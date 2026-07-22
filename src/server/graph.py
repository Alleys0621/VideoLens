"""LangGraph graph: 单节点包装 companion_chat.

agent-chat-ui 通过 LangGraph SDK 连接此 graph (langgraph dev 启动, 端口 2024).

关键帧 + 推理链放在 AIMessage.additional_kwargs (不用 custom event, 避免 API 不确定性).
前端 useKeyframeSeek 从 messages[-1].additional_kwargs 解析, 控制 video.currentTime.

video_dir 通过 LangGraph 的 configurable 注入 (前端选视频 → stream.submit config).
"""

from __future__ import annotations

import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.companion import companion_chat


class State(TypedDict):
    # add_messages reducer: 累积消息 (而不是替换), 这样 companion_node 能拿到完整 chat_history
    messages: Annotated[list, add_messages]
    video_dir: str


# keyframe 文件名格式: {scene_id}_midpoint_{seconds}s_{speaker}.jpg
_TIMESTAMP_RE = re.compile(r"_(\d+\.\d+)s_")


def _parse_timestamp(keyframe_path: str) -> float | None:
    """从 keyframe 文件名解析 timestamp.

    例: .../a0181_midpoint_0393.73s_刘梅.jpg → 393.73
    """
    m = _TIMESTAMP_RE.search(keyframe_path.replace("\\", "/"))
    return float(m.group(1)) if m else None


def _extract_msg(mc) -> tuple[str, str]:
    """从 LangChain Message 或 dict 提取 (role, content_text).

    LangGraph stream 时, messages 可能是 Message 对象 (本地 invoke) 或
    dict (HTTP API 调用, 前端 stream.submit) — 这里兼容两者.
    """
    if isinstance(mc, dict):
        mtype = mc.get("type", "")
        mrole = mc.get("role", "")
        role = "user" if (mtype == "human" or mrole == "user") else "assistant"
        content = mc.get("content", "")
    else:
        role = "user" if isinstance(mc, HumanMessage) else "assistant"
        content = mc.content
    # content 可能是 string 或 content blocks 数组
    if isinstance(content, list):
        # 提取 text blocks
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = " ".join(parts)
    elif not isinstance(content, str):
        content = str(content)
    return role, content


def _messages_to_chat_history(messages: list) -> list[dict]:
    """把 LangChain Message 列表转成 companion_chat 期望的 chat_history."""
    out = []
    for m in messages:
        role, content = _extract_msg(m)
        out.append({"role": role, "content": content})
    return out


_LLM_INSTANCE = None


def _get_streaming_llm():
    """获取 LangChain ChatOpenAI 单例 (指向 DashScope, streaming=True).

    首次调用创建并缓存, 后续复用 (预热时已创建, 问答时直接拿).
    LangGraph 自动拦截 ChatModel token 流 → messages streamMode → 前端逐字渲染.
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE
    from langchain_openai import ChatOpenAI
    from src.core.config import get_config

    cfg = get_config()
    _LLM_INSTANCE = ChatOpenAI(
        model=cfg.model_text,
        api_key=cfg.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        streaming=True,
    )
    return _LLM_INSTANCE


def _warmup():
    """启动时预热重资源 (langgraph dev import 本模块时执行).

    预热 Mem0 (向量库 + embedding client, ~5s) + LLM client (ChatOpenAI).
    第一次问答不用等懒加载. 预热失败非致命 (问答时仍会懒加载).
    """
    import os
    if os.getenv("DISABLE_WARMUP"):
        print("[warmup] 跳过 (DISABLE_WARMUP=1)", flush=True)
        return
    # 1. Mem0 (最重)
    try:
        from src.agent.mem0_client import get_memory
        get_memory()
        print("[warmup] Mem0 预热完成", flush=True)
    except Exception as e:
        print(f"[warmup] Mem0 预热失败 (非致命): {e}", flush=True)
    # 2. LLM client (ChatOpenAI 对象创建, 快)
    try:
        _get_streaming_llm()
        print("[warmup] LLM client 预热完成", flush=True)
    except Exception as e:
        print(f"[warmup] LLM client 预热失败: {e}", flush=True)
    # 2.5 LLM 意图理解预热 (消除首次 dashscope 连接冷启动 2.4s → 0.8s)
    try:
        from src.agent.intent_router import llm_route_intent
        llm_route_intent("预热", None)
        print("[warmup] 意图理解预热完成", flush=True)
    except Exception as e:
        print(f"[warmup] 意图理解预热失败 (非致命): {e}", flush=True)
    # 3. embedding 缓存 (扫描所有已建库视频, 提前建向量索引到 .npz)
    try:
        _warmup_embeddings()
    except Exception as e:
        print(f"[warmup] embedding 预热失败 (非致命): {e}", flush=True)


def _warmup_embeddings():
    """扫描 data/output/ 下所有已建库视频, 预建 embedding 缓存.

    有缓存的秒加载 (.npz), 没缓存的建 (首次 ~40s/集, 后续启动都快).
    新增集 (重新建库后) 自动补建.
    """
    import os
    from src.core.config import get_config

    cfg = get_config()
    output_root = cfg.output_root
    if not os.path.isdir(output_root):
        return

    # 扫描所有有 stage3_dryrun.json + audio.json 的视频目录
    episodes: list[str] = []
    for root, _dirs, files in os.walk(output_root):
        if "stage3_dryrun.json" in files and "audio.json" in files:
            rel = os.path.relpath(root, output_root).replace("\\", "/")
            # 跳过 _global / _batch_reports 等非视频目录
            if not rel.startswith("_") and not rel.startswith("."):
                episodes.append(rel)

    if not episodes:
        print("[warmup] 没有已建库视频, 跳过 embedding 预热", flush=True)
        return

    from src.agent.retriever import build_or_load_embeddings

    print(f"[warmup] 预热 embedding: 发现 {len(episodes)} 集", flush=True)
    for i, ep in enumerate(episodes, 1):
        try:
            build_or_load_embeddings(ep)
            print(f"[warmup] embedding {i}/{len(episodes)}: {ep} OK", flush=True)
        except Exception as e:
            print(f"[warmup] embedding {i}/{len(episodes)}: {ep} 失败: {e}", flush=True)


def companion_node(state: State, config: RunnableConfig) -> dict:
    """陪看节点: 取最后一条 user message → companion_chat → 返回 AI message."""
    import os
    from src.core.config import get_config

    messages = state["messages"]
    last = messages[-1]
    _, query = _extract_msg(last)

    # video_dir 多来源兼容: configurable (前端 config.configurable 透传) / state / state.context
    configurable = config.get("configurable", {}) if config else {}
    state_ctx = state.get("context") or {}
    if not isinstance(state_ctx, dict):
        state_ctx = {}
    video_dir = (
        configurable.get("video_dir")
        or state.get("video_dir")
        or state_ctx.get("video_dir")
        or ""
    )

    # 防御 1: 没选视频 → 友好提示 (避免 FileNotFoundError)
    if not video_dir:
        return {"messages": [AIMessage(
            content="先在左边选一集视频，我才能陪你聊剧情呀 🎬",
        )]}

    # 防御 2: 检查是否已建库
    cfg = get_config()
    dryrun_path = os.path.join(cfg.output_root, video_dir, "stage3_dryrun.json")
    if not os.path.isfile(dryrun_path):
        return {"messages": [AIMessage(
            content=f"这集（{video_dir}）还没建库呢，先跑流水线建库吧",
        )]}

    # chat_history (不含当前 query)
    chat_history = _messages_to_chat_history(messages[:-1])

    try:
        result = companion_chat(
            query=query,
            video_dir=video_dir,
            user_id=configurable.get("user_id", "default"),
            chat_history=chat_history,
            llm=_get_streaming_llm(),  # LangGraph 自动拦截 token → 前端逐字渲染
            web_search=bool(configurable.get("web_search", False)),  # 联网模式 toggle
            video_time=configurable.get("video_time"),  # 当前播放时间戳 (邻域对白检索)
        )
    except Exception as e:
        # 兜底: 任何异常都返回友好消息, 不让前端看到 raw error
        return {"messages": [AIMessage(content=f"哎呀出错了: {e}")]}

    # 关键帧 timestamp 解析 (前端用这个跳转视频)
    keyframes_meta = []
    for kf_path in result["keyframes"]:
        ts = _parse_timestamp(kf_path)
        if ts is not None:
            keyframes_meta.append({"timestamp": ts, "path": kf_path})

    ai_msg = AIMessage(
        content=result["answer"],
        additional_kwargs={
            "keyframes": keyframes_meta,
            "reasoning": result["reasoning"],
            "video_dir": video_dir,
        },
    )
    return {"messages": [ai_msg]}


def build_graph():
    """构建单节点 graph: START → companion → END."""
    g = StateGraph(State)
    g.add_node("companion", companion_node)
    g.add_edge(START, "companion")
    g.add_edge("companion", END)
    return g.compile()


# langgraph.json 引用此变量
graph = build_graph()

# 启动时预热重资源 (import 本模块时执行, 不在问答时懒加载)
_warmup()
