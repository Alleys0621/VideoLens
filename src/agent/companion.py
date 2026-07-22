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
    llm,
    query: str,
    selected: list[dict],
    video_summary: dict | None,
    arc_updates: list,
    chat_history: list[dict],
    long_term: list[str],
    video_label: str = "",
    video_context: str = "",
) -> str:
    """KB 模式: 基于检索到的 events 回答."""
    system = _get_system_prompt()
    parts = []
    if video_label:
        parts.append(f"## 当前正在看的视频\n{video_label}")
    if video_context:
        parts.append(video_context)
    if video_summary:
        parts.append(f"## 视频梗概\n{video_summary.get('episode_summary', '')}")
    evs_text = "\n\n".join(
        f"- [{e.get('event_id','')}] {e.get('title','')}\n"
        f"  角色: {', '.join(e.get('participants', []) or [])}\n"
        f"  摘要: {e.get('summary', '')}\n"
        f"  动机: {e.get('motivation', '')} ({e.get('motivation_confidence','')})\n"
        f"  结果: {e.get('outcome', '')}"
        for e in selected
    )
    parts.append(f"## 检索到的相关事件\n{evs_text}")
    if arc_updates:
        arcs = "\n".join(f"- {a.get('title','')}: {a.get('summary','')}" for a in arc_updates[:2])
        parts.append(f"## 剧情弧\n{arcs}")
    if long_term:
        mems = "\n".join(f"- {m}" for m in long_term)
        parts.append(f"## 关于这位用户我之前记得\n{mems}")
    if chat_history:
        hist = "\n".join(
            f"{'用户' if h['role']=='user' else 'Alleys'}: {h['content']}"
            for h in chat_history[-6:]
        )
        parts.append(f"## 我们刚才聊到\n{hist}")
    parts.append(
        f"## 用户现在问\n{query}\n\n"
        f"## 回答要求 (重要)\n"
        f"- 必须结合上方'我们刚才聊到'的上下文 (代词/指代关联上文, 用户说'他'你要知道指谁)\n"
        f"- 必须结合上方'关于这位用户我之前记得'的记忆 (如有)\n"
        f"- 像延续对话, 不是孤立回答\n"
        f"- 直接中文, 不超过 100 字, 像朋友聊剧, 不用标题/列表:"
    )
    return _llm_generate(
        prompt="\n\n".join(parts), system=system, stage="companion_kb",
        max_tokens=400, temperature=0.7, enable_thinking=False, llm=llm,
    )


def _llm_chitchat(
    llm, query: str, chat_history: list[dict], long_term: list[str],
    video_label: str = "",
    video_context: str = "",
) -> str:
    """闲聊模式: 人设回应 (不查 KB)."""
    system = _get_system_prompt()
    parts = []
    if video_label:
        parts.append(f"## 当前正在看的视频\n{video_label}")
    if video_context:
        parts.append(video_context)
    if long_term:
        mems = "\n".join(f"- {m}" for m in long_term)
        parts.append(f"## 关于这位用户我之前记得\n{mems}")
    if chat_history:
        hist = "\n".join(
            f"{'用户' if h['role']=='user' else 'Alleys'}: {h['content']}"
            for h in chat_history[-6:]
        )
        parts.append(f"## 我们刚才聊到\n{hist}")
    parts.append(
        f"## 用户现在说\n{query}\n\n"
        f"## 回应要求\n"
        f"- 结合上方'我们刚才聊到'的上下文延续对话\n"
        f"- 结合'关于这位用户我之前记得'的记忆 (如有)\n"
        f"- 闲聊, 像朋友, 不超过 60 字:"
    )
    result = _llm_generate(
        prompt="\n\n".join(parts), system=system, stage="companion_chitchat",
        max_tokens=200, temperature=0.8, enable_thinking=False, llm=llm,
    )
    return result or "(嗯嗯)"


def _llm_refuse(
    llm,
    query: str,
    chat_history: list[dict] | None = None,
    long_term: list[str] | None = None,
    video_label: str = "",
    video_context: str = "",
    web_search: bool = False,
) -> str:
    """拒答模式: KB 没相关内容 + 非闲聊, 诚实承认 (带上下文, 拒答也关联上文角色).
    web_search=True 时开启联网搜索 (DashScope enable_search), 允许 LLM 联网补充后回答."""
    system = _get_system_prompt()
    parts = []
    if video_label:
        parts.append(f"## 当前正在看的视频\n{video_label}")
    if video_context:
        parts.append(video_context)
    parts.append(f"用户问: 「{query}」")
    if long_term:
        mems = "\n".join(f"- {m}" for m in long_term)
        parts.append(f"## 关于这位用户我之前记得\n{mems}")
    if chat_history:
        hist = "\n".join(
            f"{'用户' if h['role']=='user' else 'Alleys'}: {h['content']}"
            for h in chat_history[-4:]
        )
        parts.append(f"## 我们刚才聊到\n{hist}")
    if web_search:
        parts.append(
            "\n这集知识库里没找到相关情节. 你可以联网搜索补充信息再回答: "
            "搜到相关内容就用Alleys口吻自然回答 (<80字, 不暴露搜索来源); "
            "搜不到或与剧集无关就诚实说不知道, 不编造:"
        )
    else:
        parts.append(
            "\n这集知识库里没找到相关情节. 用Alleys口吻诚实承认, "
            "但要关联上文 (如用户说'他', 你要知道指谁), 不编造, 30字内:"
        )
    result = _llm_generate(
        prompt="\n\n".join(parts), system=system, stage="companion_refuse",
        max_tokens=400 if web_search else 120,
        temperature=0.6, enable_thinking=False, llm=llm,
        enable_search=web_search,
    )
    return result or "这段我也没看太清诶, 你能给说说吗?"


def _llm_web_search(
    llm,
    query: str,
    video_label: str,
    chat_history: list[dict],
    long_term: list[str],
    video_context: str = "",
) -> tuple[str, list[dict]]:
    """联网搜索模式: Tavily 真实搜 query → 结果塞 prompt → LLM 基于 results 回答.

    Returns:
        (answer, web_results) — web_results 给 reasoning 展示来源 (标题+URL+摘要)。
    """
    from src.agent.web_search import tavily_search

    # 1. Tavily 搜索 (失败/空结果 → 回退诚实拒答, 不编造)
    try:
        results = tavily_search(query, max_results=5)
    except Exception as e:
        print(f"[web_search] Tavily 失败, 回退拒答: {e}", flush=True)
        return _llm_refuse(llm, query, chat_history, long_term, video_label), []
    if not results:
        return _llm_refuse(llm, query, chat_history, long_term, video_label), []

    # 2. 结果塞 prompt → LLM 回答 (不用 enable_search, 结果已在 prompt 里)
    system = _get_system_prompt()
    parts = []
    if video_label:
        parts.append(f"## 当前正在看的视频\n{video_label}")
    if video_context:
        parts.append(video_context)
    if long_term:
        mems = "\n".join(f"- {m}" for m in long_term)
        parts.append(f"## 关于这位用户我之前记得\n{mems}")
    if chat_history:
        hist = "\n".join(
            f"{'用户' if h['role']=='user' else 'Alleys'}: {h['content']}"
            for h in chat_history[-4:]
        )
        parts.append(f"## 我们刚才聊到\n{hist}")
    parts.append(f"## 用户问\n{query}")
    results_text = "\n\n".join(
        f"- [{i+1}] {r['title']}\n  {r['url']}\n  {r['content']}"
        for i, r in enumerate(results)
    )
    parts.append(f"## 联网搜索到的相关内容\n{results_text}")
    parts.append(
        "\n## 回答要求\n"
        "- 基于上方'联网搜索到的相关内容'回答\n"
        "- 用Alleys口吻, 自然口语, 不超过 120 字\n"
        "- 搜到的是剧外信息 (演员/花絮/现实) 就如实告诉用户, 不硬套剧情\n"
        "- 不暴露'根据搜索结果'这种话, 自然引用即可:"
    )
    answer = _llm_generate(
        prompt="\n\n".join(parts), system=system, stage="companion_web_search",
        max_tokens=400, temperature=0.6, enable_thinking=False, llm=llm,
    )
    return (answer or "(没搜到啥有用的)"), results


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
    # 并行: LLM 意图理解 + 检索 (两者都是 ~0.7s API 调用, 并行省一半时间)
    from src.agent.intent_router import llm_route_intent
    _intent_box: dict = {"v": "kb"}
    def _do_intent():
        _intent_box["v"] = llm_route_intent(query, video_time)
    _intent_th = threading.Thread(target=_do_intent, daemon=True)
    _intent_th.start()

    bm25_hits = index.search(query, top_k=5) if index else []
    # BM25 top_score 用于意图分流 (KB_SCORE_THRESHOLD=2.0) + reasoning 展示
    top_score = float(bm25_hits[0][1]) if bm25_hits else 0.0

    # 向量检索 (和意图理解并行; 总是全集检索, 意图结果后决定用不用)
    try:
        from src.agent.retriever import (
            build_or_load_embeddings, embed_query, vector_search, rrf_fuse,
        )
        events_emb, segs_emb, _ev_text, _sg_text = build_or_load_embeddings(video_dir)
        query_emb = embed_query(query)
        events_vec_hits = vector_search(query_emb, events_emb, top_k=5)
        events_fused = rrf_fuse(bm25_hits, events_vec_hits, top_k=5)
        segs_vec_hits = vector_search(query_emb, segs_emb, top_k=5)
        matched_segments = [
            segments[i] for i, _ in segs_vec_hits[:5] if i < len(segments)
        ]
    except Exception as e:
        print(f"[retriever] 向量检索失败, 回退纯 BM25: {e}", flush=True)
        events_fused = bm25_hits  # fallback

    # 等意图理解完成 (和检索并行了, 通常同时完成, join 几乎不等)
    _intent_th.join()
    intent_type = _intent_box["v"]

    # 指代类 (deictic): 清空向量 segments, 只用时间邻域 (避免全集其他时间相同词干扰)
    if intent_type == "deictic" and video_time is not None and video_time >= 0:
        matched_segments = []

    retrieved = [
        {"event_id": events[i].get("event_id", ""),
         "title": events[i].get("title", ""),
         "score": round(float(s), 3)}
        for i, s in events_fused
    ]
    selected = [events[i] for i, _ in events_fused[:3]]

    t_retrieval = time.time()  # 混合检索完成 (打点)

    # 3. 意图分流 (语义路由 intent_type + BM25 阈值)
    if intent_type == "meta" and video_summary:
        intent = "kb_meta"
        selected = []  # 元问题不用 events, 只用 video_summary
    elif intent_type == "chitchat":
        intent = "chitchat"
    else:
        intent = _classify_intent(top_score)  # kb/refuse (BM25 阈值)

    # 联网搜索时, refuse → web_search (前端推理卡片区分展示 "联网搜索了")
    if web_search and intent == "refuse":
        intent = "web_search"

    # 4. 按意图处理 (llm 优先 streaming, 否则 QwenTextClient fallback)
    long_term = search_relevant_memories(user_id, query, top_k=3)
    # 当前视频身份 (作品 · 季 · 集), 注入 prompt 让 Alleys 知道用户在看什么
    video_label = " · ".join(p for p in video_dir.split("/") if p)

    # 当前画面上下文 (时间戳 + 邻域对白), 让 Alleys 知道用户此刻看到什么
    # 总是注入, LLM 自己判断 query 是否和当前画面相关 (用户问的不一定和画面有关)
    video_context = ""
    if video_time is not None and video_time >= 0 and segments:
        nearby = _retrieve_segments_by_time(segments, float(video_time), window=15)
        if nearby:
            lines = "\n".join(
                f"[{float(s.get('begin_time') or 0):.0f}s] {s.get('speaker_pred','?')}: {s.get('text','')}"
                for s in nearby
            )
            video_context = f"## 用户当前画面 (约 {float(video_time):.0f}s, 附近对白)\n{lines}"

    # 语义相关的对白 (向量检索 segments 命中, 解决 BM25 漏的同义/改述召回)
    if matched_segments:
        seg_lines = "\n".join(
            f"[{float(s.get('begin_time') or 0):.0f}s] {s.get('speaker_pred','?')}: {s.get('text','')}"
            for s in matched_segments
        )
        video_context += f"\n\n## 语义相关的对白 (检索命中)\n{seg_lines}"

    if intent in ("kb", "kb_meta"):
        answer = _llm_kb_answer(llm, query, selected, video_summary, arc_updates,
                                chat_history, long_term, video_label, video_context)
        keyframes = _events_to_keyframes(selected, scenes)
        evidence = []
        for e in selected:
            for ev in e.get("evidence", []) or []:
                evidence.append({
                    "event_id": e.get("event_id"),
                    "description": ev.get("description", ""),
                    "scene_ids": ev.get("scene_ids", []),
                })
        # 存长期记忆
        _async_add_memory(user_id, query, answer, video_dir)
    elif intent == "chitchat":
        answer = _llm_chitchat(llm, query, chat_history, long_term, video_label, video_context)
        keyframes = []
        evidence = []
        _async_add_memory(user_id, query, answer, video_dir)
    else:  # refuse
        keyframes = []
        evidence = []
        if web_search:
            # 联网模式: Tavily 真实搜索 → 结果塞 prompt → LLM 基于 results 回答
            answer, web_results = _llm_web_search(
                llm, query, video_label, chat_history, long_term, video_context
            )
            _async_add_memory(user_id, query, answer, video_dir)
        else:
            answer = _llm_refuse(llm, query, chat_history, long_term, video_label, video_context)
            # 拒答不存记忆 (避免误导未来检索)

    t_end = time.time()
    reasoning = {
        "intent": intent,
        "query": query,
        "top_score": round(top_score, 3),
        "threshold": KB_SCORE_THRESHOLD,
        "retrieved": retrieved,
        "selected": [{"event_id": e.get("event_id"), "title": e.get("title"),
                      "participants": e.get("participants", []),
                      "summary": e.get("summary", "")} for e in selected],
        "evidence": evidence,
        "web_results": web_results,
        "timings": {
            "retrieval_ms": round((t_retrieval - t_start) * 1000),
            "llm_ms": round((t_end - t_retrieval) * 1000),
            "total_ms": round((t_end - t_start) * 1000),
        },
    }
    # 全链路打点 (后端)
    print(f"[companion] query={query!r} intent={intent} top_score={round(top_score, 2)} | "
          f"retrieval={reasoning['timings']['retrieval_ms']}ms | "
          f"llm={reasoning['timings']['llm_ms']}ms | total={reasoning['timings']['total_ms']}ms",
          flush=True)

    return {"answer": answer, "reasoning": reasoning, "keyframes": keyframes}
