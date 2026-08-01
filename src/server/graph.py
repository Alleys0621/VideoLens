"""LangGraph graph: 单节点包装 companion_chat (对话为核心架构).

agent-chat-ui 通过 LangGraph SDK 连接此 graph (langgraph dev 启动, 端口 2024).
关键帧 + 推理链放在 AIMessage.additional_kwargs, 前端 useKeyframeSeek 解析.
video_dir 通过 configurable 注入.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.companion import companion_chat
from src.core.logging import get_logger

logger = get_logger()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    video_dir: str


def _extract_msg(mc) -> tuple[str, str]:
    """从 LangChain Message 或 dict 提取 (role, content_text)."""
    if isinstance(mc, dict):
        mtype = mc.get("type", "")
        mrole = mc.get("role", "")
        role = "user" if (mtype == "human" or mrole == "user") else "assistant"
        content = mc.get("content", "")
    else:
        role = "user" if isinstance(mc, HumanMessage) else "assistant"
        content = mc.content
    if isinstance(content, list):
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
    out = []
    for m in messages:
        role, content = _extract_msg(m)
        out.append({"role": role, "content": content})
    return out


def companion_node(state: State, config) -> dict:
    """陪看节点: 取最后一条 user message → companion_chat → AI message."""
    import re
    from langchain_core.messages import AIMessage

    messages = state["messages"]
    last = messages[-1]
    _, query = _extract_msg(last)

    configurable = config.get("configurable", {}) if config else {}
    video_dir = (
        configurable.get("video_dir")
        or state.get("video_dir")
        or ""
    )

    if not video_dir:
        return {"messages": [AIMessage(content="先在左边选一集视频,我才能陪你聊剧情呀 🎬")]}

    chat_history = _messages_to_chat_history(messages[:-1])

    try:
        result = companion_chat(
            query=query,
            video_dir=video_dir,
            user_id=configurable.get("user_id", "default"),
            chat_history=chat_history,
            web_search=bool(configurable.get("web_search", False)),
            video_time=configurable.get("video_time"),
        )
    except Exception as e:
        logger.exception("[graph] companion_chat failed")
        return {"messages": [AIMessage(content=f"哎呀出错了: {e}")]}

    # keyframes timestamp 解析 (前端跳转用)
    ts_re = re.compile(r"_(\d+\.\d+)s_")
    keyframes_meta = []
    for kf_path in result["keyframes"]:
        m = ts_re.search(kf_path.replace("\\", "/"))
        if m:
            keyframes_meta.append({"timestamp": float(m.group(1)), "path": kf_path})

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
