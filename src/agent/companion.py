"""对话为核心的陪看架构 (AlleysVid 唯一主入口).

核心设计 (经过 E1-E5 实验验证):
  - 每轮都检索 KB, 但 prompt 引导"按需引用" (避免 context pollution 又保证事实正确率 ~95%)
  - L1 + L2 画像始终注入 user prompt
  - 5 轮短期记忆始终注入
  - Mem0 长期记忆始终注入
  - 主 LLM 默认 flash (qwen3.7-flash), COMPANION_MAIN_MODEL=plus 可切回
  - 联网开启时走 Tavily

核心: agent 是用户朋友, 以对话为主, KB 仅在需要时作为参考.

返回: {answer, reasoning, keyframes}
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from src.core.config import get_config
from src.core.helpers.json_utils import load_json
from src.core.logging import get_logger

logger = get_logger()


# ============================================================
# 主 LLM (固定一个模型, 不按意图切)
# 默认 flash (测速度/质量), 可用 COMPANION_MAIN_MODEL=plus 切回 plus
# ============================================================

_LLM_POOL: dict[str, ChatOpenAI] = {}

_MAIN_MODEL_KEY = "main"


def _get_main_llm() -> tuple[ChatOpenAI, str]:
    """单例 ChatOpenAI. 返回 (llm, model_key).

    模型选择 (env COMPANION_MAIN_MODEL):
      - 未设置 / 'flash' → cfg.model_text_flash (默认, 快)
      - 'plus'          → cfg.model_text (质量高, 慢)
    """
    import os as _os
    override = _os.getenv("COMPANION_MAIN_MODEL", "").strip().lower()
    model_key = "plus" if override == "plus" else "flash"

    if model_key in _LLM_POOL:
        return _LLM_POOL[model_key], model_key

    cfg = get_config()
    model_name = cfg.model_text if model_key == "plus" else cfg.model_text_flash
    _LLM_POOL[model_key] = ChatOpenAI(
        model=model_name,
        api_key=cfg.dashscope_api_key,
        base_url=cfg.dashscope_base_url,
        streaming=True,
    )
    return _LLM_POOL[model_key], model_key


# ============================================================
# 数据加载 (跟 companion.py 一致, 但简化)
# ============================================================

def _load_video_data(video_dir: str) -> tuple[list, list, list, list]:
    """加载一集的 events / actions / scenes / segments.

    复用 video_utils.load_episode_data (走 stage3_kb.json + visual.json + audio.json).
    """
    from src.agent.video_utils import load_episode_data
    return load_episode_data(video_dir)


def _load_video_summary(video_dir: str) -> dict | None:
    """加载视频梗概 (如果有)."""
    cfg = get_config()
    p = os.path.join(cfg.output_root, video_dir, "video_summary.json")
    if not os.path.isfile(p):
        return None
    return load_json(p)


# ============================================================
# 检索 (跟 companion.py 一致: BM25 + 向量 + RRF + 时间 boost)
# ============================================================

def _retrieve(
    query: str,
    video_dir: str,
    events: list,
    scenes: list,
    segments: list,
    video_time: float | None,
    top_k: int = 3,
) -> tuple[list[tuple[int, float]], list[dict], float]:
    """混合检索 + 时间 boost. 返回 (events_fused, selected, top_score).

    复用 companion.py 的 BM25Index (跟 v1 同源, 避免 import rank_bm25).
    """
    from src.agent.retriever import (
        build_or_load_embeddings, embed_query, vector_search,
        rrf_fuse, apply_time_boost,
    )
    from src.eval.stage3_retrieval import BM25Index, build_searchable_text

    # BM25 (跟 companion.py 一致)
    index = BM25Index([build_searchable_text(e) for e in events]) if events else None
    bm25_hits = index.search(query, top_k=5) if index else []
    top_score = float(bm25_hits[0][1]) if bm25_hits else 0.0

    # 向量检索
    events_emb, _segs_emb, _et, _st = build_or_load_embeddings(video_dir)
    query_emb = embed_query(query)
    events_vec_hits = vector_search(query_emb, events_emb, top_k=5)

    # RRF + 时间 boost
    events_fused = rrf_fuse(bm25_hits, events_vec_hits, top_k=5)
    events_fused = apply_time_boost(events_fused, events, scenes, video_time)

    selected = [events[i] for i, _ in events_fused[:top_k]]
    return events_fused, selected, top_score


# ============================================================
# Prompt 构建 (核心: 对话为核心)
# ============================================================

def _build_system_prompt() -> str:
    """读 yaml 的 Alleys 人设 (companion_prompts.system)."""
    cfg = get_config()
    prompts = cfg.prompts.get("companion_prompts", {}) or {}
    return (prompts.get("system") or "").strip()


def _format_video_kb(selected: list[dict]) -> str:
    """格式化 video_kb 检索结果 (作为'参考', 不是必须用)."""
    if not selected:
        return "(本集知识库未检索到强相关内容, 你可以基于对话上下文自由回应)"
    parts = []
    for e in selected:
        title = e.get("title", "")
        participants = ", ".join(e.get("participants", []) or [])
        summary = e.get("summary", "")
        line = f"- [{title}]"
        if participants:
            line += f" 角色: {participants}"
        if summary:
            line += f"\n  摘要: {summary}"
        parts.append(line)
    return "\n".join(parts)


def _format_working_memory(video_time: float | None, events: list, actions: list) -> str:
    """按 video_time 找当前 event 的工作记忆. 真·间隙/无 video_time 返回空串 (跳过该块)."""
    if video_time is None or not events or not actions:
        return ""
    aid2t: dict[str, tuple[float, float]] = {}
    for a in actions:
        ts = (a.get("evidence") or {}).get("timestamp")
        if ts and len(ts) == 2:
            try:
                aid2t[a.get("action_id")] = (float(ts[0]), float(ts[1]))
            except (TypeError, ValueError):
                continue
    if not aid2t:
        return ""
    evt_ranges: list[tuple[float, float, dict]] = []
    for e in events:
        ets = [aid2t[x["action_id"]] for x in (e.get("actions") or []) if x.get("action_id") in aid2t]
        if ets:
            evt_ranges.append((min(t[0] for t in ets), max(t[1] for t in ets), e))
    if not evt_ranges:
        return ""
    hit = next((r for r in evt_ranges if r[0] <= video_time <= r[1]), None)
    if hit is None:
        hit = min(evt_ranges, key=lambda r: min(abs(video_time - r[0]), abs(video_time - r[1])))
        gap = min(abs(video_time - hit[0]), abs(video_time - hit[1]))
        if gap > 5:
            return ""
    start, end, e = hit
    title = (e.get("title") or "").strip()
    summary = (e.get("summary") or "").strip()
    if not summary:
        return ""
    return f"{title}\n{summary}\n（大概在 {start:.0f}~{end:.0f} 秒）"


def _format_short_term_memory(chat_history: list[dict], n: int = 10) -> str:
    """short_term_memory: 最近 n 条消息 (一轮 = 一问一答 = 2 条, 默认 10 条 = 5 轮)."""
    if not chat_history:
        return "(刚开始聊)"
    lines = []
    for h in chat_history[-n:]:
        role = "用户" if h.get("role") == "user" else "Alleys"
        lines.append(f"{role}: {h.get('content', '')}")
    return "\n".join(lines)


def _format_l1_l2(user_profile: dict | None, show_profile: dict | None) -> str:
    """L1/L2 画像 → 行为指令式文本. L1 决定"怎么回", L2 决定"回什么".

    返回空串则 profile 块自动跳过 (companion.py 拼装逻辑会过滤空 content).
    """
    if not user_profile:
        return ""

    L: list[str] = ["[user_profile · 以下指令覆盖 persona 默认行为]"]

    # ---- 表现层: 用户当下指令, 不门控, 排第一位 (跨会话即时生效) ----
    attitude = (user_profile.get("alleys_attitude") or "").strip()
    if attitude:
        L.append(f"【即时·用户直接指令】{attitude}")

    pref = user_profile.get("alleys_response_preference")
    pref_map = {
        "反问引导": "多反问推进(你觉得呢/猜下一步)",
        "直接表态": "少反问,直接给观点",
        "拱火加码": "用户吐槽时跟着加码,把感受说得更具体",
        "冷静降温": "用户激动时拉回客观,不跟着嗨",
    }
    if pref in pref_map:
        L.append(f"【即时·回应策略】{pref_map[pref]}")

    # ---- 内核层: conf_stable>=0.6 才注入 (稳定人格, 宁慢勿噪) ----
    conf_stable = float(user_profile.get("conf_stable", 0) or 0)
    if conf_stable >= 0.6:
        style = user_profile.get("interaction_style")
        style_map = {
            "吐槽型": "用户爱吐槽",
            "分析型": "用户爱分析剧情逻辑",
            "陪伴型": "用户重情绪交流",
            "提问型": "用户靠提问推进",
        }
        if style in style_map:
            L.append(style_map[style])

        initiative = user_profile.get("interaction_initiative")
        if initiative == "被动型":
            L.append("用户被动,平淡段你主动起话题(预测/挑细节),用户在说时别打断")
        elif initiative == "主动型":
            L.append("用户主动起话题,你跟进即可,不必硬找话题")

        humor = user_profile.get("humor_level")
        teasing = user_profile.get("teasing_tolerance")
        if humor == "高":
            tag = {"能被吐槽": "可反吐槽用户",
                   "只能吐槽剧情": "只吐槽剧情",
                   "完全不能": "但别吐槽任何人"}.get(teasing or "", "只吐槽剧情")
            L.append(f"接梗强,{tag}")
        elif humor == "低":
            L.append("接梗弱,少开玩笑正经聊")

        peeves = [p for p in (user_profile.get("pet_peeves") or []) if p][:4]
        if peeves:
            L.append(f"雷区(遇到可主动吐槽):{'、'.join(peeves)}")

        sp = user_profile.get("spoiler_tolerance")
        if sp == "谨慎":
            L.append("剧透谨慎,只聊已看过部分")
        elif sp == "拒绝":
            L.append("不剧透,只聊当前画面")

    # 只有头标签没有内容 → 返回空, 让拼装跳过
    if len(L) == 1:
        return ""

    # ---- L2: 回什么 ----
    if show_profile and float(show_profile.get("confidence", 0) or 0) >= 0.5:
        fav = [c for c in (show_profile.get("favorite_characters") or []) if c][:3]
        if fav:
            L.append(f"心头好(戏份相关可主动提):{'、'.join(fav)}")
        disliked = [d for d in (show_profile.get("disliked_elements") or []) if d][:3]
        if disliked:
            L.append(f"本剧雷区:{'、'.join(disliked)}")
        ops = show_profile.get("character_opinions") or []
        if isinstance(ops, list):
            for o in ops[:3]:
                if isinstance(o, dict):
                    ch = o.get("character", "")
                    op = o.get("opinion", "")
                    if ch and op:
                        L.append(f"  · {ch}: {op}")

    return "\n".join(L)


def _format_episode_memory(memories: list[str]) -> str:
    """episode_memory: Mem0 跨会话长期记忆."""
    if not memories:
        return "(暂无长期记忆)"
    return "\n".join(f"- {m}" for m in memories[:3])


def _build_user_prompt(
    query: str,
    video_label: str,
    chat_history: list[dict],
    user_profile: dict | None,
    show_profile: dict | None,
    long_term: list[str],
    selected: list[dict],
    web_results_text: str = "",
    working_memory: str = "",
) -> str:
    """构建 user prompt: 画像 + 记忆 + KB(参考) + query.

    完整提示词在 yaml `companion_prompts` 里定义 (含每段标题/引导语/回答要求),
    这里只负责取内容 → 跳过空段 → 填 `{content}` 占位符 → 按 yaml 顺序拼接.
    改 yaml 即可热编辑, 不用动代码.
    """
    cfg = get_config()
    prompts = cfg.prompts.get("companion_prompts", {}) or {}
    blocks = prompts.get("user_blocks", {}) or {}
    tail = prompts.get("user_tail", "")

    # 各段内容 (key 对应 yaml user_blocks 的块名)
    contents: dict[str, str] = {
        "working_memory": working_memory,
        "user_profile": _format_l1_l2(user_profile, show_profile),
        "short_term_memory": _format_short_term_memory(chat_history, n=10),
        "episode_memory": _format_episode_memory(long_term),
        "video_kb": _format_video_kb(selected),
        "web_search": web_results_text,
    }

    # 按 yaml 顺序拼接非空块
    parts: list[str] = []
    for key, block in blocks.items():
        if not isinstance(block, dict):
            continue
        template = block.get("template", "")
        if not template:
            continue
        content = contents.get(key, "")
        if "{content}" in template:
            if not content:
                continue
            parts.append(template.replace("{content}", content))
        else:
            parts.append(template)
    body = "\n\n".join(parts)

    if tail:
        return f"{body}\n\n{tail.replace('{query}', query)}"
    return body


# ============================================================
# 主入口
# ============================================================

def companion_chat(
    *,
    query: str,
    video_dir: str,
    user_id: str = "default",
    chat_history: list[dict] | None = None,
    web_search: bool = False,
    video_time: float | None = None,
) -> dict:
    """对话为核心的陪看主入口.

    返回 {answer, reasoning, keyframes} 跟 companion_chat 一致.
    """
    t_start = time.time()
    chat_history = chat_history or []
    cfg = get_config()
    video_label = video_dir

    # 1. 加载数据
    events, actions, scenes, segments = _load_video_data(video_dir)
    video_summary = _load_video_summary(video_dir)
    show = video_dir.split("/")[0] if video_dir else ""

    # 2. L1 / L2 画像
    from src.agent.profile_store import (
        load_user_profile, load_show_profile,
    )
    from src.agent.video_utils import load_watching_state
    user_profile = None
    show_profile = None
    if user_id and user_id != "default":
        user_profile = load_user_profile(user_id)
        if show:
            show_profile = load_show_profile(user_id, show)
        # watching_state: 前端持续上报的最新坐标, 优先于 configurable 快照
        watching = load_watching_state(user_id)
        if watching and watching.get("video_time") is not None:
            video_time = watching["video_time"]

    # 3. 检索 (KB) - 每轮都跑, 给 LLM 当参考
    events_fused, selected, top_score = _retrieve(
        query, video_dir, events, scenes, segments, video_time, top_k=3,
    )

    # 4. Mem0 长期记忆
    long_term: list[str] = []
    try:
        from src.agent.mem0_client import search_relevant_memories
        long_term = search_relevant_memories(user_id, query, top_k=3)
    except Exception as e:
        logger.warning(f"[companion] mem0 search 失败: {e}")

    # 5. 联网 (开启时)
    web_results_text = ""
    web_results: list[dict] = []
    if web_search:
        try:
            from src.agent.web_search import tavily_search
            results = tavily_search(query, max_results=5)
            if results:
                web_results = results
                web_results_text = "\n\n".join(
                    f"- [{i+1}] {r['title']}\n  {r['url']}\n  {r['content']}"
                    for i, r in enumerate(results)
                )
        except Exception as e:
            logger.warning(f"[companion] tavily 失败: {e}")

    t_retrieval = time.time()

    # 6. 构建 prompt
    working_memory = _format_working_memory(video_time, events, actions)
    system = _build_system_prompt()
    user_prompt = _build_user_prompt(
        query=query,
        video_label=video_label,
        chat_history=chat_history,
        user_profile=user_profile,
        show_profile=show_profile,
        long_term=long_term,
        selected=selected,
        web_results_text=web_results_text,
        working_memory=working_memory,
    )

    # 打全量 prompt 方便排查 (system / user_prompt 各一条, 用分隔符包起来便于 grep)
    logger.info(f"[prompt] === SYSTEM ===\n{system}")
    logger.info(f"[prompt] === USER ===\n{user_prompt}")

    # 7. 调主 LLM (固定模型, 真·streaming — token 级推到 LangGraph → SDK → 前端)
    from langchain_core.messages import SystemMessage, HumanMessage
    llm, _main_model_key = _get_main_llm()
    extra_body = {"enable_thinking": False}
    if web_search:
        extra_body["enable_search"] = True
    bound = llm.bind(max_tokens=400, temperature=0.7, extra_body=extra_body)
    msgs = [SystemMessage(content=system), HumanMessage(content=user_prompt)]
    pieces: list[str] = []
    t_first_token: float | None = None
    for chunk in bound.stream(msgs):
        if t_first_token is None:
            t_first_token = time.time()
        c = getattr(chunk, "content", "") or ""
        if isinstance(c, str):
            pieces.append(c)
    answer = "".join(pieces).strip()

    t_end = time.time()

    # 8. 异步写 Mem0 + 触发画像更新 (复用 video_utils 的异步逻辑)
    try:
        from src.agent.video_utils import async_add_memory, async_maybe_update_profile
        async_add_memory(user_id, query, answer, video_dir)
        if user_id and user_id != "default":
            async_maybe_update_profile(user_id, chat_history, query, answer, show=show)
    except Exception as e:
        logger.warning(f"[companion] async side effects 失败: {e}")

    # 9. keyframes (基于检索 + 当前时间)
    keyframes: list[str] = []
    if selected:
        try:
            from src.agent.video_utils import events_to_keyframes
            keyframes = events_to_keyframes(selected, scenes)
        except Exception:
            pass

    # 10. reasoning
    retrieved = [
        {"event_id": events[i].get("event_id", ""),
         "title": events[i].get("title", ""),
         "score": round(float(s), 3)}
        for i, s in events_fused
    ]
    reasoning = {
        "architecture": "v2_dialog_first",
        "main_llm_model": _main_model_key,
        "query": query,
        "top_score": round(top_score, 3),
        "retrieved": retrieved,
        "selected": [{"event_id": e.get("event_id"), "title": e.get("title"),
                      "participants": e.get("participants", []),
                      "summary": e.get("summary", "")} for e in selected],
        "user_profile_loaded": user_profile is not None,
        "show_profile_loaded": show_profile is not None,
        "long_term_count": len(long_term),
        "web_results": web_results,
        "timings": {
            "retrieval_ms": round((t_retrieval - t_start) * 1000),
            "llm_ms": round((t_end - t_retrieval) * 1000),
            "first_token_ms": round((t_first_token - t_retrieval) * 1000) if t_first_token else None,
            "total_ms": round((t_end - t_start) * 1000),
        },
    }

    logger.info(
        f"[companion] query={query!r} top_score={round(top_score, 2)} "
        f"selected={len(selected)} long_term={len(long_term)} "
        f"profile={user_profile is not None}/{show_profile is not None} | "
        f"retrieval={reasoning['timings']['retrieval_ms']}ms | "
        f"first_token={reasoning['timings']['first_token_ms']}ms | "
        f"llm={reasoning['timings']['llm_ms']}ms | "
        f"total={reasoning['timings']['total_ms']}ms | "
        f"answer={answer!r}"
    )

    return {"answer": answer, "reasoning": reasoning, "keyframes": keyframes}
