"""上下文白名单 (Context Budget).

核心设计原则:
    Context is a privilege, not a default.
    用户没问剧情时, 不要把剧情/检索/记忆塞给 LLM — 否则 Alleys 会"自作聪明"复述剧情.

按 intent 任务决定哪些 context 段允许进入 prompt:

    chitchat   : 只 persona + video_label + 最近 2 轮 (不注入剧情/检索/记忆)
    companion  : persona + video_label + focus_character + 最近 3 轮 (接情绪, 不主动讲剧情)
    deictic    : persona + video_label + current_event + keyframe + 邻域对白 + 最近 2 轮
    knowledge  : persona + video_label + retrieval + 最近 4 轮 (可给 1 条记忆)
    meta       : persona + video_label + video_summary + 最近 2 轮
    refuse     : persona + video_label + 最近 2 轮 (web_search 分支额外注入搜索结果)

低置信度意图 → chitchat 行 (最小上下文). 见 intent_router.IntentResult.safe_task.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# 派生: 当前画面上下文 (无状态, 纯函数)
# ============================================================

def scene_at_time(scenes: list[dict], t: float) -> dict | None:
    """按时间戳找到当前画面所属的 scene."""
    if t is None or t < 0:
        return None
    for s in scenes:
        try:
            st = float(s.get("start_time", 0) or 0)
            ed = float(s.get("end_time", 0) or 0)
        except (TypeError, ValueError):
            continue
        if st <= t <= ed:
            return s
    return None


def scene_keyframe_at_time(scenes: list[dict], t: float) -> str:
    """当前时间对应 scene 的主 keyframe (deictic 兜底用)."""
    s = scene_at_time(scenes, t)
    if not s:
        return ""
    kfs = s.get("keyframe_paths") or []
    return kfs[0] if kfs else ""


def derive_watching_context(
    video_time: float | None,
    events: list[dict],
    scenes: list[dict],
    segments: list[dict],
    window: float = 15.0,
) -> dict:
    """从当前播放时间派生轻量画面上下文 (无持久化).

    Returns:
        {
          "focus_character": str | None,   # 邻域对白里高频 speaker
          "current_event":   dict | None,  # 当前时间所属的 event (deictic 用)
          "current_scene_id": str | None,
          "current_keyframe": str,         # 当前 scene 的主 keyframe (deictic 兜底)
        }
    """
    empty = {
        "focus_character": None,
        "current_event": None,
        "current_scene_id": None,
        "current_keyframe": "",
    }
    if video_time is None or video_time < 0:
        return empty

    t = float(video_time)

    # 1. 邻域对白 → focus_character (高频 speaker)
    lo, hi = t - window, t + window
    speaker_counts: dict[str, int] = {}
    for s in segments:
        try:
            begin = float(s.get("begin_time") or 0)
            end = float(s.get("end_time") or 0)
        except (TypeError, ValueError):
            continue
        if begin <= hi and end >= lo:
            sp = (s.get("speaker_pred") or "").strip()
            if sp and not sp.lower().startswith("char_unknown"):
                speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
    focus_character = (
        max(speaker_counts.items(), key=lambda x: x[1])[0]
        if speaker_counts else None
    )

    # 2. 当前 scene → 找到所属 event
    cur_scene = scene_at_time(scenes, t)
    cur_scene_id = cur_scene.get("scene_id") if cur_scene else None
    cur_keyframe = (cur_scene.get("keyframe_paths") or [""])[0] if cur_scene else ""
    current_event = None
    if cur_scene_id:
        for e in events:
            for ev in e.get("evidence", []) or []:
                if cur_scene_id in (ev.get("scene_ids") or []):
                    current_event = e
                    break
            if current_event:
                break

    return {
        "focus_character": focus_character,
        "current_event": current_event,
        "current_scene_id": cur_scene_id,
        "current_keyframe": cur_keyframe,
    }


# ============================================================
# 上下文段渲染 (公共格式化)
# ============================================================

def render_history(chat_history: list[dict], n: int) -> str:
    """最近 n 轮对话 (user/Alleys)."""
    if not chat_history:
        return ""
    lines = []
    for h in chat_history[-n:]:
        role = "用户" if h.get("role") == "user" else "Alleys"
        lines.append(f"{role}: {h.get('content', '')}")
    return "\n".join(lines)


def _format_event(e: dict) -> str:
    parts = [f"[{e.get('event_id', '')}] {e.get('title', '')}"]
    parts.append(f"  角色: {', '.join(e.get('participants', []) or [])}")
    if e.get("summary"):
        parts.append(f"  摘要: {e['summary']}")
    if e.get("motivation"):
        parts.append(f"  动机: {e['motivation']}")
    if e.get("outcome"):
        parts.append(f"  结果: {e['outcome']}")
    return "\n".join(parts)


def _format_events(events: list[dict]) -> str:
    return "\n\n".join(_format_event(e) for e in events)


# 每种 task 允许注入的 context 段 (白名单)
CONTEXT_BUDGET: dict[str, set[str]] = {
    "chitchat":  {"video_label", "history_2"},
    "companion": {"video_label", "history_3", "focus_character", "long_term_1", "show_affinity"},
    "deictic":   {"video_label", "history_2", "focus_character", "current_event", "nearby_dialogue"},
    "knowledge": {"video_label", "history_4", "retrieval", "long_term_1", "show_affinity"},
    "meta":      {"video_label", "history_2", "video_summary"},
    "refuse":    {"video_label", "history_2"},
}


def build_context_sections(
    task: str,
    *,
    video_label: str = "",
    watching: dict | None = None,
    video_context: str = "",
    long_term: list[str] | None = None,
    selected: list[dict] | None = None,
    video_summary: dict | None = None,
    chat_history: list[dict] | None = None,
    web_results_text: str = "",
    show_profile: dict | None = None,
) -> list[tuple[str, str]]:
    """按 task 白名单产出 prompt context 段 (title, body).

    未命中白名单的段一律不进 prompt. 不确定的 task 走 chitchat 行 (最小上下文).
    """
    budget = CONTEXT_BUDGET.get(task, CONTEXT_BUDGET["chitchat"])
    watching = watching or {}
    sections: list[tuple[str, str]] = []

    if "video_label" in budget and video_label:
        sections.append(("当前正在看的视频", video_label))

    if "focus_character" in budget and watching.get("focus_character"):
        sections.append((
            "当前画面焦点角色",
            f"约在此刻画面里: {watching['focus_character']}",
        ))

    if "current_event" in budget and watching.get("current_event"):
        sections.append((
            "当前事件 (用户此刻画面所属)",
            _format_event(watching["current_event"]),
        ))

    if "nearby_dialogue" in budget and video_context:
        # deictic: video_context 里只有邻域对白 (companion.py 已清掉语义 segments)
        sections.append(("用户当前画面附近对白", video_context))

    if "retrieval" in budget and selected:
        sections.append(("检索到的相关事件", _format_events(selected)))

    if "video_summary" in budget and video_summary:
        summary_text = video_summary.get("episode_summary", "")
        if summary_text:
            sections.append(("视频梗概", summary_text))

    if "long_term_1" in budget and long_term:
        # 只给 1 条, 避免记忆喧宾夺主
        mem = long_term[0]
        if mem:
            sections.append(("关于这位用户我之前记得", f"- {mem}"))

    if "show_affinity" in budget and show_profile:
        aff = _format_show_affinity(show_profile)
        if aff:
            sections.append(("这位用户在这部剧里的偏好", aff))

    if chat_history:
        n = 4 if "history_4" in budget else 3 if "history_3" in budget else 2
        hist = render_history(chat_history, n)
        if hist:
            sections.append(("我们刚才聊到", hist))

    # web_results 只给 refuse+web_search 分支 (由调用方传入非空文本)
    if web_results_text:
        sections.append(("联网搜索到的相关内容", web_results_text))

    return sections


def _format_show_affinity(profile: dict) -> str:
    """渲染 L2: 喜欢的角色 + 关注角色 + 评价 + 主题偏好 + 不喜欢."""
    parts = []
    fav = [c for c in (profile.get("favorite_characters") or []) if c]
    if fav:
        parts.append("喜欢: " + "、".join(fav[:5]))
    att = [c for c in (profile.get("attention_characters") or []) if c]
    if att:
        parts.append("关注: " + "、".join(att[:5]))
    ops = profile.get("character_opinions") or []
    if isinstance(ops, list):
        op_lines = []
        for o in ops[:5]:
            if not isinstance(o, dict):
                continue
            ch = o.get("character", "")
            opinion = o.get("opinion", "")
            if ch and opinion:
                op_lines.append(f"  - {ch}: {opinion}")
        if op_lines:
            parts.append("角色评价:\n" + "\n".join(op_lines))
    themes = [t for t in (profile.get("theme_preferences") or []) if t]
    if themes:
        parts.append("偏好主题: " + "、".join(themes[:5]))
    disliked = [d for d in (profile.get("disliked_elements") or []) if d]
    if disliked:
        parts.append("不喜欢: " + "、".join(disliked[:5]))
    return "\n".join(parts)


def sections_to_prompt(sections: list[tuple[str, str]]) -> str:
    """把 context 段渲染成 prompt 文本块."""
    return "\n\n".join(f"## {t}\n{b}" for t, b in sections if b)
