"""陪看智能体 v2: 意图分流 + 帧级定位 + 推理过程可视化.

入口: companion_chat(query, video_dir, user_id, chat_history)
返回: {answer, reasoning, keyframes}

4 功能:
  1. 帧级定位 — Event.evidence.scene_ids → visual.scenes.keyframe_paths
  2. 推理可视化 — reasoning 结构化 (意图 / 检索 top-K / 选中 / 证据)
  3. 意图分流 + 拒答 — BM25 score + 闲聊规则 → KB / chitchat / refuse 三分支 (暂不联网)
  4. 人设回复 — Alleys人设 + 情绪感知 (prompt 在 yaml::companion_xiaoying_system)

角色归一 / BM25 检索逻辑从 scripts/frontend_app.py 迁移, frontend_app 后续改为 import 本模块.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime

from src.core.config import get_config
from src.core.helpers.json_utils import load_json
from src.core.llm.qwen_text import QwenTextClient
from src.eval.stage3_retrieval import BM25Index, build_searchable_text
from src.agent.mem0_client import search_relevant_memories, add_conversation_memory
from src.agent.context_builder import (
    build_context_sections,
    sections_to_prompt,
    derive_watching_context,
    scene_keyframe_at_time,
)
from src.agent.profile_store import (
    load_user_profile,
    render_profile_overlay,
    increment_message_counter,
    load_show_profile,
    PROFILE_UPDATE_THRESHOLD,
)
from src.agent.profile_updater import maybe_update_user_profile, maybe_update_show_profile


# ============================================================
# 常量
# ============================================================

# BM25 score 阈值: top1 >= 此值走 KB 模式 (经验值, 第一季数据 top1~9, 不相关 <3)
# 降到 2.0: 让泛化/弱相关查询也走 KB (避免频繁拒答), LLM 自己判断相关性
KB_SCORE_THRESHOLD = 2.0
# 意图理解 (deictic/meta/chitchat) → 语义路由 (intent_router.py), 不再用正则.
# kb/refuse 仍用 BM25 top_score 阈值判断, 后续可改 LLM-based.

# 角色归一 (从 frontend_app.py 迁移)
_PINYIN_MAP = {
    "xue": "夏雪", "mei": "刘梅", "xing": "刘星", "yu": "夏雨", "donghai": "夏东海",
    "liumei": "刘梅", "liu_mei": "刘梅", "lumei": "刘梅",
    "liu_xing": "刘星", "liuxing": "刘星",
    "xiaxue": "夏雪", "xia_xue": "夏雪",
    "xiayu": "夏雨", "xia_yu": "夏雨",
    "xiadonghai": "夏东海", "xia_donghai": "夏东海", "xiaodonghai": "夏东海", "xiadh": "夏东海",
    "grandma": "姥姥", "grandpa": "爷爷", "mother": "玛丽", "father": "父亲",
}
_UNKNOWN_PREFIXES = ("char_unknown", "char_passerby", "char_luren", "char_lu_ren", "char_new", "路人", "新角色")

# 人设 fallback (yaml 没加载时用)
_XIAOYING_FALLBACK = (
    "你是「Alleys」, 一个陪看搭子智能体, 25 岁女生, 陪伴用户一起看剧聊剧情. "
    "活泼温暖, 用大白话讲, 答案不超过 80 字. 不知道就诚实说不知道, 不编造."
)


# ============================================================
# 数据加载
# ============================================================

def _load_episode_data(video_dir: str) -> tuple[list, list, list, list]:
    """加载一集的 events / actions / scenes / audio_segments.

    Returns:
        (events, actions, scenes, segments) —
        events/actions 来自 stage3_dryrun.json,
        scenes 来自 visual.json,
        segments 来自 audio.json (Stage 1 ASR, 含 begin_time/end_time/speaker_pred/text).
    """
    cfg = get_config()
    ep_dir = os.path.join(cfg.output_root, video_dir)
    dryrun_path = os.path.join(ep_dir, "stage3_dryrun.json")
    visual_path = os.path.join(ep_dir, "visual.json")
    audio_path = os.path.join(ep_dir, "audio.json")

    if not os.path.isfile(dryrun_path):
        raise FileNotFoundError(f"未建库: {dryrun_path}; 先跑 Stage 1-3")

    dryrun = load_json(dryrun_path)
    events = dryrun.get("events", []) or []
    actions = dryrun.get("actions", []) or []

    scenes = []
    if os.path.isfile(visual_path):
        scenes = load_json(visual_path).get("scenes", []) or []

    segments = []
    if os.path.isfile(audio_path):
        segments = load_json(audio_path).get("segments", []) or []

    return events, actions, scenes, segments


def _retrieve_segments_by_time(
    segments: list, video_time: float, window: float = 15.0, max_n: int = 8,
) -> list[dict]:
    """基于视频时间戳检索邻域 audio segments (用户当前画面附近的对白).

    用于回答 "这一刻在说什么 / 这个'哎'称呼谁" 等指代当前画面的提问.
    返回 video_time ± window 秒内的 segments (按时间序).
    """
    if not segments or video_time is None or video_time < 0:
        return []
    lo = video_time - window
    hi = video_time + window
    hits = []
    for s in segments:
        begin = float(s.get("begin_time") or 0)
        end = float(s.get("end_time") or 0)
        # segment 时间窗和 [lo, hi] 有重叠
        if begin <= hi and end >= lo:
            hits.append(s)
    return hits[:max_n]


def _build_char_map() -> dict[str, str]:
    """从 _global/characters.json 建角色归一映射 (character_id + aliases → name)."""
    cfg = get_config()
    p = os.path.join(cfg.output_root, "_global", "characters.json")
    if not os.path.isfile(p):
        return {}
    chars = load_json(p)
    m = {}
    for c in chars:
        name = c.get("name", "")
        if not name:
            continue
        m[c.get("character_id", "")] = name
        for alias in c.get("aliases", []) or []:
            m[alias] = name
        m[name] = name
    return m


def _normalize_participant(p: str, char_map: dict[str, str]) -> str:
    """把 participants/targets 里的 char_id / 拼音 / 别名 归一成标准中文名."""
    if not p:
        return ""
    p = p.strip()
    if p in char_map:
        return char_map[p]
    for prefix in _UNKNOWN_PREFIXES:
        if p.startswith(prefix) or p == prefix:
            return "未知角色"
    if p.startswith("char_"):
        key = p[5:].lower()
        if key in _PINYIN_MAP:
            return _PINYIN_MAP[key]
        last = key.rsplit("_", 1)[-1]
        if last in _PINYIN_MAP:
            return _PINYIN_MAP[last]
        no_us = key.replace("_", "")
        if no_us in _PINYIN_MAP:
            return _PINYIN_MAP[no_us]
        return "未知角色"
    return p


def _normalize_events(events: list[dict], char_map: dict[str, str]) -> list[dict]:
    """归一 events 的 participants + actions[].target."""
    out = []
    for e in events:
        ne = dict(e)
        ne["participants"] = [_normalize_participant(p, char_map) for p in (e.get("participants") or [])]
        new_actions = []
        for a in e.get("actions", []) or []:
            na = dict(a)
            t = a.get("target")
            if isinstance(t, str):
                na["target"] = _normalize_participant(t, char_map)
            elif isinstance(t, list):
                na["target"] = [_normalize_participant(x, char_map) for x in t]
            new_actions.append(na)
        ne["actions"] = new_actions
        out.append(ne)
    return out


# ============================================================
# 帧级定位
# ============================================================

def _scene_id_to_keyframe(scene_id: str, scenes: list) -> str:
    """scene_id → 主 keyframe 路径."""
    for s in scenes:
        if s.get("scene_id") == scene_id:
            kfps = s.get("keyframe_paths") or []
            if kfps:
                return kfps[0]
    return ""


def _events_to_keyframes(events: list[dict], scenes: list, max_frames: int = 3) -> list[str]:
    """从 events 的 evidence.scene_ids 提取关键帧路径 (去重, 限 max_frames)."""
    frames = []
    seen = set()
    for e in events:
        for ev in e.get("evidence", []) or []:
            for sid in ev.get("scene_ids", []) or []:
                if sid in seen:
                    continue
                kf = _scene_id_to_keyframe(sid, scenes)
                if kf:  # 路径存在性由 UI 端检查 (避免每次 stat)
                    frames.append(kf)
                    seen.add(sid)
                    if len(frames) >= max_frames:
                        return frames
    return frames


# ============================================================
# 意图分流
# ============================================================

def _classify_intent(top_score: float) -> str:
    """kb/refuse 二分类 (chitchat 由语义路由 intent_router 预判)."""
    if top_score >= KB_SCORE_THRESHOLD:
        return "kb"
    return "refuse"


# ============================================================
# LLM 回答 (三分支)
# ============================================================

def _get_system_prompt() -> str:
    """从 yaml 加载Alleys人设, 追加当前时间, 失败用 fallback."""
    cfg = get_config()
    p = cfg.prompts.get("companion_xiaoying_system", {})
    if isinstance(p, dict):
        base = p.get("user", "") or _XIAOYING_FALLBACK
    else:
        base = _XIAOYING_FALLBACK
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    now = datetime.now()
    now_str = f"{now.strftime('%Y-%m-%d')} {weekdays[now.weekday()]} {now.strftime('%H:%M:%S')}"
    return f"{base}\n\n## 当前时间\n{now_str}"


def _llm_generate(
    prompt: str,
    system: str,
    stage: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
    llm=None,
    enable_search: bool = False,
) -> str:
    """统一 LLM 调用: 优先用 llm (LangChain ChatOpenAI — LangGraph 自动拦截 token 流,
    通过 messages streamMode 推给前端实现逐字渲染); llm 为 None 时 fallback 到 QwenTextClient (非流式).
    enable_search=True 时开启 DashScope 原生联网 (qwen-plus 等支持, LLM 自动判断是否触发搜索)."""
    if llm is not None:
        from langchain_core.messages import SystemMessage, HumanMessage
        extra_body: dict = {"enable_thinking": enable_thinking}
        if enable_search:
            extra_body["enable_search"] = True
        bound = llm.bind(
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        resp = bound.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        return (resp.content or "").strip()
    # fallback: 非流式 (直连 DashScope)
    raw = QwenTextClient().generate(
        prompt=prompt, system=system, stage=stage,
        max_tokens=max_tokens, temperature=temperature,
        enable_thinking=enable_thinking,
    )
    return (raw or "").strip()


def _llm_kb_answer(
    *args, **kwargs,
) -> str:
    """已废弃: 上下文拼装统一迁到 context_builder.build_context_sections."""
    raise RuntimeError("_llm_kb_answer 已废弃, 请用 _llm_answer + build_context_sections")


def _llm_chitchat(*args, **kwargs) -> str:
    raise RuntimeError("_llm_chitchat 已废弃, 请用 _llm_answer + build_context_sections")


def _llm_refuse(*args, **kwargs) -> str:
    raise RuntimeError("_llm_refuse 已废弃, 请用 _llm_answer + build_context_sections")


def _llm_web_search(*args, **kwargs) -> tuple[str, list[dict]]:
    raise RuntimeError("_llm_web_search 已废弃, 请用 _llm_answer + build_context_sections")


# ============================================================
# 统一回复: context 段 + 任务策略 → 单次 streaming LLM
# ============================================================

# task → (max_tokens, temperature)
_TASK_PARAMS: dict[str, tuple[int, float]] = {
    "chitchat": (200, 0.7),
    "companion": (260, 0.7),
    "deictic": (360, 0.6),
    "knowledge": (400, 0.7),
    "meta": (500, 0.6),
    "refuse": (160, 0.6),
}


def _requirements(task: str, web: bool = False) -> str:
    """按 task 给最终回复的硬要求 (含字数上限)."""
    if web:
        return (
            "- 基于上方'联网搜索到的相关内容'回答\n"
            "- 用Alleys口吻, 自然口语, 不超过 120 字\n"
            "- 搜到的是剧外信息 (演员/花絮/现实) 就如实告诉用户, 不硬套剧情\n"
            "- 不暴露'根据搜索结果'这种话, 自然引用即可:"
        )
    return {
        "chitchat": (
            "- 闲聊, 像朋友, 不超过 60 字\n"
            "- 你没在看画面: 不要说\"正看到/刚看到/画面里\", 也不要描述当前剧情进度\n"
            "- 用户说卡了/好了/暂停就事论事回 (如\"嗯, 等你好\"), 不借机演剧情\n"
            "- 不每句都\"哈哈\"开头, 没情绪信号就正常说话\n"
            "- 不解释自己为什么这么说, 禁止\"我寻思/逗你乐/脑补/想多了\"这类自我旁白:"
        ),
        "companion": (
            "- 陪着看, 接住用户情绪, 不超过 80 字\n"
            "- 可以提焦点角色, 但不要描述画面/表情/动作 (你没在看)\n"
            "- 不复述/预测剧情细节, 不自我旁白, 不\"太真实了/绝了/有那味儿\"这种套话:"
        ),
        "deictic": (
            "- 基于当前事件 / 邻域对白回答, 不超过 100 字\n"
            "- 指代要说清 (谁在说话/这一幕在干什么)\n"
            "- 不要延伸到未来剧情:"
        ),
        "knowledge": (
            "- 基于检索到的相关事件回答, 不超过 100 字\n"
            "- 不延伸到未来剧情, 不剧透\n"
            "- 像朋友聊剧, 不用标题/列表:"
        ),
        "meta": (
            "- 基于视频梗概概括, 不超过 150 字\n"
            "- 不剧透后续集:"
        ),
        "refuse": (
            "- 用Alleys口吻诚实承认, 关联上文, 不编造, 40 字内:"
        ),
    }.get(task, "- 直接中文回答, 不超过 80 字:")


def _llm_answer(
    llm,
    task: str,
    query: str,
    sections: list[tuple[str, str]],
    emotion: str = "neutral",
    user_state: str = "无明显状态",
    web: bool = False,
    system_overlay: str = "",
) -> str:
    """统一回复: 拼好 context + query + 状态参考 + requirements → 单次 streaming LLM.

    task 决定 max_tokens/temperature/requirements; sections 已由白名单裁剪过.
    emotion/user_state 来自用户理解模块, 仅作一行参考, 不污染 context 白名单.
    system_overlay 来自 L1 用户画像 (style 类), 拼到 system prompt 末尾, 自然影响语气.
    """
    system = _get_system_prompt()
    if system_overlay:
        system = f"{system}\n\n{system_overlay}"
    ctx = sections_to_prompt(sections)
    # 状态参考: 只在有信号时给 (neutral + 无明显状态 时不注入, 避免噪音)
    state_hint = ""
    if emotion != "neutral" or user_state not in ("", "无明显状态"):
        state_hint = (
            f"\n\n（用户状态参考：{user_state} · 情绪 {emotion}。"
            "仅供参考，结合原话判断，不要生硬贴标签，也不要直接复述这句话。）"
        )
    prompt = (
        f"{ctx}\n\n"
        f"## 用户现在问\n{query}{state_hint}\n\n"
        f"## 回答要求\n{_requirements(task, web=web)}"
    )
    max_tokens, temperature = _TASK_PARAMS.get(task, (300, 0.7))
    stage = f"companion_{task}" + ("_web" if web else "")
    result = _llm_generate(
        prompt=prompt,
        system=system,
        stage=stage,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=False,
        llm=llm,
        # 注意: web 分支的搜索结果已在 prompt 里, 不再开 enable_search
        enable_search=False,
    )
    if task == "chitchat":
        return result or "(嗯嗯)"
    if task == "refuse" and not web:
        return result or "这段我也没看太清诶, 你能给说说吗?"
    return result or ""


# ============================================================
# 主入口
# ============================================================

def _async_add_memory(user_id: str, query: str, answer: str, video_id: str) -> None:
    """异步写 Mem0 记忆 (线程, 不阻塞 companion 返回 — Mem0 写入慢, 避免拖累响应)."""
    def _write():
        try:
            add_conversation_memory(user_id, query, answer, video_id=video_id)
        except Exception as e:
            print(f"[Mem0] async add 失败: {e}", flush=True)

    threading.Thread(target=_write, daemon=True).start()


def _async_maybe_update_profile(
    user_id: str, chat_history: list[dict], query: str, answer: str, show: str = "",
) -> None:
    """异步累加对话计数, 达阈值触发 L1 + L2 画像增量更新 (不阻塞回复)."""
    def _run():
        try:
            n = increment_message_counter(user_id)
            if n >= PROFILE_UPDATE_THRESHOLD:
                hist = list(chat_history or []) + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ]
                maybe_update_user_profile(user_id, hist)
                if show:
                    maybe_update_show_profile(user_id, show, hist)
        except Exception as e:
            print(f"[profile] async trigger 失败: {e}", flush=True)

    threading.Thread(target=_run, daemon=True).start()


def companion_chat(
    query: str,
    video_dir: str,
    user_id: str = "default",
    chat_history: list[dict] | None = None,
    llm=None,
    web_search: bool = False,
    video_time: float | None = None,
) -> dict:
    """陪看智能体主入口.

    Args:
        query: 用户问题
        video_dir: 视频目录 (如 "家有儿女/第一季/第01集")
        user_id: 用户标识 (Mem0 用)
        chat_history: 短期对话历史 [{role, content}, ...]
        llm: LangChain ChatModel (如 ChatOpenAI); 提供时走 streaming (LangGraph 自动拦截 token),
             None 时 fallback 到 QwenTextClient 非流式

    Returns:
        {answer, reasoning, keyframes}
        - reasoning: {intent, query, retrieved, selected, evidence}
        - keyframes: 要展示的帧路径列表 (帧级定位)
    """
    chat_history = chat_history or []
    import time
    t_start = time.time()

    # 1. 加载数据 + 归一
    events_raw, _actions, scenes, segments = _load_episode_data(video_dir)
    char_map = _build_char_map()
    events = _normalize_events(events_raw, char_map)

    # video_summary / arc_updates (可选, 单集 KB 文件)
    cfg = get_config()
    ep_dir = os.path.join(cfg.output_root, video_dir)
    kb_path = os.path.join(ep_dir, "stage3_kb.json")
    video_summary = None
    arc_updates = []
    if os.path.isfile(kb_path):
        kb = load_json(kb_path)
        video_summary = kb.get("video_summary")
        arc_updates = kb.get("arc_updates", []) or []

    # 2. 混合检索: BM25 (events) + 向量 (events + segments) → RRF 融合
    index = BM25Index([build_searchable_text(e) for e in events]) if events else None
    retrieved = []
    top_score = 0.0
    selected = []
    web_results: list[dict] = []
    matched_segments: list[dict] = []  # 向量检索命中的对白 (对白级, 注入 prompt)
    # 并行: LLM 用户理解 (qwen-max, task+emotion+user_state) + 检索 (两者 ~0.7s, 并行省一半)
    from src.agent.intent_router import llm_route_intent, IntentResult
    video_label = " · ".join(p for p in video_dir.split("/") if p)

    # L1 用户长期画像 → system overlay (confidence 够才注入, style 类, 不进 context)
    # L2 作品画像 → context 段 (companion/knowledge 才注入, 见 context_builder 白名单)
    show = video_dir.split("/")[0] if video_dir else ""
    profile_overlay = ""
    show_profile = None
    if user_id and user_id != "default":
        profile_overlay = render_profile_overlay(load_user_profile(user_id))
        if show:
            show_profile = load_show_profile(user_id, show)

    _intent_box: dict = {
        "v": IntentResult(
            task="kb", task_confidence=0.0,
            emotion="neutral", emotion_confidence=0.0,
            user_state="无明显状态",
        ),
    }
    def _do_intent():
        _intent_box["v"] = llm_route_intent(
            query, video_time, video_label=video_label, chat_history=chat_history,
        )
    _intent_th = threading.Thread(target=_do_intent, daemon=True)
    _intent_th.start()

    bm25_hits = index.search(query, top_k=5) if index else []
    # BM25 top_score 仅用于 reasoning 展示 + knowledge 兜底判空
    top_score = float(bm25_hits[0][1]) if bm25_hits else 0.0

    # 向量检索 (和意图理解并行; knowledge 才会进 prompt, 但先算着, 意图回来再决定用不用)
    try:
        from src.agent.retriever import (
            build_or_load_embeddings, embed_query, vector_search, rrf_fuse,
        )
        events_emb, segs_emb, _ev_text, _sg_text = build_or_load_embeddings(video_dir)
        query_emb = embed_query(query)
        events_vec_hits = vector_search(query_emb, events_emb, top_k=5)
        events_fused = rrf_fuse(bm25_hits, events_vec_hits, top_k=5)
    except Exception as e:
        print(f"[retriever] 向量检索失败, 回退纯 BM25: {e}", flush=True)
        events_fused = bm25_hits  # fallback

    # 等意图理解完成 (和检索并行了, 通常同时完成, join 几乎不等)
    _intent_th.join()
    intent_result: IntentResult = _intent_box["v"]
    # safe_*: 低置信度保守回退 (task→chitchat, emotion→neutral)
    task = intent_result.safe_task
    emotion = intent_result.safe_emotion
    user_state = intent_result.user_state

    retrieved = [
        {"event_id": events[i].get("event_id", ""),
         "title": events[i].get("title", ""),
         "score": round(float(s), 3)}
        for i, s in events_fused
    ]
    selected = [events[i] for i, _ in events_fused[:3]]

    t_retrieval = time.time()  # 混合检索完成 (打点)

    # 3. 任务纠偏: 某些任务在缺数据时退化到更诚实的任务
    # meta 没梗概 / knowledge 没检索到 → refuse (不编造)
    if task == "meta" and not video_summary:
        task = "refuse"
    if task == "knowledge" and not selected:
        task = "refuse"

    # 4. 上下文 (按 task 白名单最小化注入)
    # video_label 已在理解模块调用前算好

    # 当前画面派生 (无状态): focus_character / current_event / current_keyframe
    watching = derive_watching_context(video_time, events, scenes, segments)

    # video_context 只在 deictic 时算 (邻域对白). 其他任务不注入画面上下文.
    video_context = ""
    if task == "deictic" and video_time is not None and video_time >= 0 and segments:
        nearby = _retrieve_segments_by_time(segments, float(video_time), window=15)
        if nearby:
            lines = "\n".join(
                f"[{float(s.get('begin_time') or 0):.0f}s] {s.get('speaker_pred','?')}: {s.get('text','')}"
                for s in nearby
            )
            video_context = f"约 {float(video_time):.0f}s, 附近对白:\n{lines}"

    # 长期记忆只在 knowledge/companion 时取 (其他任务不注入, 避免寒暄时被记忆带偏)
    long_term: list[str] = []
    if task in ("knowledge", "companion"):
        try:
            long_term = search_relevant_memories(user_id, query, top_k=1)
        except Exception as e:
            print(f"[mem0] search 失败: {e}", flush=True)

    # web_search 分支: refuse + 用户开了联网 → Tavily
    web = bool(web_search) and task == "refuse"
    web_results: list[dict] = []
    web_results_text = ""
    if web:
        try:
            from src.agent.web_search import tavily_search
            results = tavily_search(query, max_results=5)
        except Exception as e:
            print(f"[web_search] Tavily 失败, 回退拒答: {e}", flush=True)
            results = []
        if results:
            web_results = results
            web_results_text = "\n\n".join(
                f"- [{i+1}] {r['title']}\n  {r['url']}\n  {r['content']}"
                for i, r in enumerate(results)
            )
        else:
            web = False  # 没搜到, 走普通 refuse

    sections = build_context_sections(
        task,
        video_label=video_label,
        watching=watching,
        video_context=video_context,
        long_term=long_term,
        selected=selected,
        video_summary=video_summary,
        chat_history=chat_history,
        web_results_text=web_results_text,
        show_profile=show_profile,
    )

    answer = _llm_answer(
        llm, task, query, sections,
        emotion=emotion, user_state=user_state, web=web,
        system_overlay=profile_overlay,
    )

    # 5. keyframes / evidence (按 task)
    evidence: list[dict] = []
    if task == "deictic":
        cur_event = watching.get("current_event")
        keyframes = _events_to_keyframes([cur_event], scenes) if cur_event else []
        if not keyframes and watching.get("current_keyframe"):
            keyframes = [watching["current_keyframe"]]
        if cur_event:
            for ev in cur_event.get("evidence", []) or []:
                evidence.append({
                    "event_id": cur_event.get("event_id"),
                    "description": ev.get("description", ""),
                    "scene_ids": ev.get("scene_ids", []),
                })
    elif task == "knowledge":
        keyframes = _events_to_keyframes(selected, scenes)
        for e in selected:
            for ev in e.get("evidence", []) or []:
                evidence.append({
                    "event_id": e.get("event_id"),
                    "description": ev.get("description", ""),
                    "scene_ids": ev.get("scene_ids", []),
                })
    else:
        keyframes = []

    # 6. 记忆写入策略
    # chitchat/companion/knowledge/web → 写; refuse (非 web) → 不写 (避免误导未来检索)
    if task in ("chitchat", "companion", "knowledge") or web:
        _async_add_memory(user_id, query, answer, video_dir)

    # 7. L1/L2 画像: 非 refuse 对话累计计数, 达阈值异步增量更新 (不阻塞回复)
    if user_id and user_id != "default" and task != "refuse":
        _async_maybe_update_profile(user_id, chat_history, query, answer, show=show)

    t_end = time.time()
    reasoning = {
        "intent": task,
        "intent_raw": intent_result.task,
        "task_confidence": round(intent_result.task_confidence, 2),
        "emotion": emotion,
        "emotion_raw": intent_result.emotion,
        "emotion_confidence": round(intent_result.emotion_confidence, 2),
        "user_state": user_state,
        "query": query,
        "top_score": round(top_score, 3),
        "threshold": KB_SCORE_THRESHOLD,
        "retrieved": retrieved,
        "selected": [{"event_id": e.get("event_id"), "title": e.get("title"),
                      "participants": e.get("participants", []),
                      "summary": e.get("summary", "")} for e in selected],
        "watching": {
            "focus_character": watching.get("focus_character"),
            "current_event_id": (watching.get("current_event") or {}).get("event_id"),
            "current_scene_id": watching.get("current_scene_id"),
        },
        "evidence": evidence,
        "web_results": web_results,
        "timings": {
            "retrieval_ms": round((t_retrieval - t_start) * 1000),
            "llm_ms": round((t_end - t_retrieval) * 1000),
            "total_ms": round((t_end - t_start) * 1000),
        },
    }
    # 全链路打点 (后端)
    print(f"[companion] query={query!r} task={task}(raw={intent_result.task},c={intent_result.task_confidence:.2f}) "
          f"emo={emotion}(raw={intent_result.emotion},c={intent_result.emotion_confidence:.2f}) "
          f"top_score={round(top_score, 2)} focus={watching.get('focus_character')} | "
          f"retrieval={reasoning['timings']['retrieval_ms']}ms | "
          f"llm={reasoning['timings']['llm_ms']}ms | total={reasoning['timings']['total_ms']}ms",
          flush=True)

    return {"answer": answer, "reasoning": reasoning, "keyframes": keyframes}
