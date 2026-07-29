"""语义意图路由 (Semantic Routing).

预定义 intent 的语义描述 (自然语言, 不是关键词).
query embedding 和 intent 描述 embedding 余弦匹配 → 最相关 intent.
纯语义理解, 不用正则/关键词. 用户没说过那些字符也能匹配到同一意图.

扩展新 intent: 在 INTENT_CATALOGUE 加描述 + companion.py 加对应处理.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.logging import get_logger

logger = get_logger()


# ============================================================
# 意图目录: intent_name → {descriptions, desc}
# descriptions 是"用户可能的问法"的自然语言描述 (多条, 取平均向量)
# embedding 捕获语义变体 — 用户换种说法也能匹配到同一 intent
# ============================================================

INTENT_CATALOGUE: dict[str, dict] = {
    "chitchat": {
        "descriptions": [
            "用户打招呼, 表达情绪, 闲聊",
            "你好, 谢谢, 辛苦了, 加油",
            "哈哈, 嘿嘿, 拜拜, 早安晚安",
            "情绪表达, 鼓励, 感谢, 问候",
            "用户问你是谁, 你会什么, 今天天气",
            "用户问 Alleys 的爱好, 喜不喜欢看剧, 喜欢看什么",
            "你喜欢看电视剧吗, 你喜欢看什么类型的剧",
            "日常寒暄, 不针对当前剧情内容提问",
        ],
        "desc": "闲聊 → 不查 KB, 直接人设回应",
    },
    "companion": {
        "descriptions": [
            "用户对剧情或角色随口反应, 吐槽, 发表感想",
            "这段太好笑了, 我好喜欢这个角色, 真感人",
            "演员演得不错, 这个角色好讨厌",
            "跟着剧情一起感叹, 不是明确提问",
        ],
        "desc": "陪伴型反应 → 可给少量剧情上下文, 不重检索",
    },
    "deictic": {
        "descriptions": [
            "用户指着当前视频画面提问, 问这一刻发生了什么",
            "这个画面里谁在说话, 这个动作是什么意思",
            "刚才那个镜头, 此刻的对白, 这一刻的表情",
            "视频里这个人是谁, 这个场景在干什么",
            "暂停在这里问画面内容, 这一句台词",
            "这一幕在演什么, 这个角色现在在干嘛",
        ],
        "desc": "指代当前画面 → 只用时间邻域检索 (跳过全集向量)",
    },
    "knowledge": {
        "descriptions": [
            "用户明确问剧情, 角色, 因果关系",
            "刘星考试考了多少分, 夏雪为什么闹别扭",
            "这一集主要冲突是什么, 夏东海怎么教育刘星",
            "后面会发生什么, 这个角色结局如何",
            "谁做了什么, 为什么这样做, 这件事的前因后果",
        ],
        "desc": "剧情知识 → 检索 events + segments 回答",
    },
    "meta": {
        "descriptions": [
            "用户问这集整体讲了什么, 剧情梗概",
            "这集的故事, 这集的主要内容是什么",
            "概括一下这集, 这集的总结",
            "这集在讲什么, 大概什么剧情",
        ],
        "desc": "元问题 → 用 video_summary 回答 (不用 events 检索)",
    },
    "refuse": {
        "descriptions": [
            "用户问剧外信息, 演员现实身份, 拍摄花絮, 制作背景",
            "演员叫什么名字, 导演是谁, 这部剧什么时候拍的, 在哪里拍的",
            "推荐其他剧, 问怎么买票, 问外部世界知识或新闻",
            "询问与当前视频剧情完全无关的现实信息",
        ],
        "desc": "拒答 → 不查 KB, 礼貌说明",
    },
}


# 预计算的 intent 向量 (预热时算一次, 后续复用)
_intent_vectors: dict[str, np.ndarray] = {}


def build_intent_vectors() -> dict[str, np.ndarray]:
    """预热: 每个 intent 的多条描述 → embedding → 取平均作为 intent 向量.

    只算一次 (模块级缓存), 后续 route_intent 直接用.
    """
    global _intent_vectors
    if _intent_vectors:
        return _intent_vectors
    from src.agent.retriever import _embed_texts

    for intent, info in INTENT_CATALOGUE.items():
        embs = _embed_texts(info["descriptions"])
        _intent_vectors[intent] = embs.mean(axis=0)  # 多条描述取平均
    logger.info(
        f"[intent_router] 预建 {len(_intent_vectors)} 个语义意图向量: "
        f"{list(_intent_vectors.keys())}"
    )
    return _intent_vectors


def route_intent(
    query_emb: np.ndarray,
    threshold: float = 0.55,
) -> tuple[str | None, float]:
    """语义路由: query → intent (embedding 余弦匹配).

    Args:
        query_emb: 已计算的 query embedding (复用混合检索的, 零额外 API 调用)
        threshold: 最低余弦分数, 低于此返回 None (走默认 kb 处理)

    Returns:
        (intent_name, score) — intent_name 为 None 表示不确定, 走默认 kb.
        score ∈ [0, 1] (余弦相似度).
    """
    vectors = build_intent_vectors()
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    best_intent: str | None = None
    best_score = 0.0
    for intent, vec in vectors.items():
        v = vec / (np.linalg.norm(vec) + 1e-8)
        score = float(q @ v)
        if score > best_score:
            best_score = score
            best_intent = intent
    if best_score >= threshold:
        return best_intent, best_score
    return None, best_score  # 不确定, 走默认


def fast_route_intent(
    query: str,
    query_emb: np.ndarray | None = None,
    threshold: float = 0.55,
) -> IntentResult:
    """纯 embedding 任务路由 (Path 1): 只决定 task, emotion/user_state 用兜底.

    若 query_emb 未提供, 会现场调用 embedding API (多一次调用, 测试时尽量复用).
    低于 threshold 时 task 回退为 kb (companion.py 中 safe_task 会再转成 chitchat).
    """
    if query_emb is None:
        from src.agent.retriever import embed_query
        query_emb = embed_query(query)

    task, score = route_intent(query_emb, threshold=threshold)
    if task is None:
        task = "kb"
    # embedding 路由的 task_confidence 用余弦分; 低于 0.6 时 safe_task 会回退 chitchat
    return IntentResult(
        task=task,
        task_confidence=round(score, 3),
        emotion="neutral",
        emotion_confidence=0.0,
        user_state="无明显状态",
    )


def hybrid_route_intent(
    query: str,
    query_emb: np.ndarray,
    fallback_threshold: float | None = None,
    video_time: float | None = None,
    video_label: str = "",
    chat_history: list[dict] | None = None,
) -> IntentResult:
    """Path 1 + fallback: embedding 先路由, 置信度低时 fallback 到 LLM.

    工业常用模式: embedding first → confidence? → 高置信度直通, 低置信度走 LLM.
    fallback_threshold 默认 None → 读 cfg.hybrid_threshold (pipeline.yaml).
    阈值依据见 scripts/probe_threshold_sweep.py (50 条混合数据, 0.65 时准确率 96%).
    """
    if fallback_threshold is None:
        from src.core.config import get_config
        fallback_threshold = get_config().hybrid_threshold
    emb_result = fast_route_intent(query, query_emb=query_emb)
    # task=="kb" 时 score 必 < fast_route_intent.threshold (0.55) < fallback_threshold,
    # 前半个条件已过滤, 无需再判 task.
    if emb_result.task_confidence >= fallback_threshold:
        return emb_result
    return llm_route_intent(
        query, video_time, video_label=video_label, chat_history=chat_history
    )


# ============================================================
# LLM 意图理解 (qwen-turbo, 真正的语义理解)
# ============================================================

@dataclass
class IntentResult:
    """用户理解模块的输出 (一次强模型 LLM 调用).

    两件独立的事, 置信度分开:
      - task / task_confidence: 决定 Alleys 这次回复能看到哪些上下文 (白名单).
      - emotion / emotion_confidence: 决定语气/共情, 但不参与上下文路由.
    user_state: 一句话描述「当前文本能直接看出的状态/诉求」, 仅作为最终回复的参考,
                不进 context_sections, 不一定可信 → 最终 LLM 仍自行判断.

    低置信度的处理 (保守):
      - task_confidence 低  → safe_task = chitchat (最小上下文, 不乱喂剧情).
      - emotion_confidence 低 → safe_emotion = neutral (不强行共情).
    """

    task: str
    task_confidence: float
    emotion: str
    emotion_confidence: float
    user_state: str

    @property
    def safe_task(self) -> str:
        return "chitchat" if self.task_confidence < INTENT_CONFIDENCE_THRESHOLD else self.task

    @property
    def safe_emotion(self) -> str:
        if self.emotion_confidence < EMOTION_CONFIDENCE_THRESHOLD:
            return "neutral"
        return self.emotion if self.emotion in _VALID_EMOTIONS else "neutral"


# 置信度阈值: 任一低于此值都走保守回退
INTENT_CONFIDENCE_THRESHOLD = 0.6
EMOTION_CONFIDENCE_THRESHOLD = 0.6

# 兜底 (LLM 调用失败 / JSON 解析失败): task_confidence=0 → safe_task=chitchat
_FALLBACK = IntentResult(
    task="kb",
    task_confidence=0.0,
    emotion="neutral",
    emotion_confidence=0.0,
    user_state="无明显状态",
)

_VALID_TASKS = {"chitchat", "companion", "deictic", "knowledge", "meta", "refuse"}
_VALID_EMOTIONS = {
    "joy", "surprised", "curious", "confused", "sad", "angry", "frustrated", "neutral",
}


def _parse_task(raw: str) -> str:
    """从 LLM 输出里提取合法 task, 非法→kb (走 safe 后是 chitchat)."""
    s = (raw or "").strip().lower().rstrip("。.,，")
    mapping = [
        ("deictic", ["deictic", "指代", "画面", "当前", "这一刻", "这一幕"]),
        ("meta", ["meta", "梗概", "整集", "概括", "这集讲"]),
        ("chitchat", ["chitchat", "闲聊", "聊天", "问候"]),
        ("companion", ["companion", "吐槽", "陪伴", "感想"]),
        ("refuse", ["refuse", "拒答", "超出", "剧外"]),
        ("knowledge", ["knowledge", "知识", "剧情", "为什么", "角色"]),
    ]
    for norm, keys in mapping:
        if any(k in s for k in keys):
            return norm
    return "kb"


def _parse_emotion(raw: str) -> str:
    """LLM 输出 → 合法 emotion, 非法→neutral."""
    s = (raw or "").strip().lower()
    mapping = {
        "joy": ["joy", "开心", "被逗笑", "高兴", "兴奋"],
        "surprised": ["surprised", "意外", "惊讶"],
        "curious": ["curious", "好奇"],
        "confused": ["confused", "疑惑", "困惑", "懵"],
        "sad": ["sad", "难过", "难受", "心疼", "低落"],
        "angry": ["angry", "生气", "气死", "愤怒"],
        "frustrated": ["frustrated", "无奈", "失望", "憋屈", "无语"],
        "neutral": ["neutral", "无明显", "平静"],
    }
    for norm, keys in mapping.items():
        if any(k in s for k in keys):
            return norm
    return "neutral"


def llm_route_intent(
    query: str,
    video_time: float | None = None,
    video_label: str = "",
    chat_history: list[dict] | None = None,
) -> IntentResult:
    """强模型语义理解 (模型由 cfg.model_intent 决定, 默认 qwen3.7-flash): 一次调用输出 task / emotion / user_state.

    和检索并行, 不增加额外延迟. 返回 IntentResult, 字段独立:
      - task/task_confidence: 驱动上下文白名单 (Context Budget).
      - emotion/emotion_confidence: 共情/语气线索 (不路由).
      - user_state: 仅根据当前文本能直接看出的状态/诉求, 给最终回复参考.

    设计原则: 不确定就保守 (task→chitchat, emotion→neutral).
    """
    time_str = f"{video_time:.0f}s" if video_time is not None else "未知"

    # 最近 1~2 轮对话, 解决「他/那个」指代 (无历史则省略)
    history_block = ""
    if chat_history:
        recent = chat_history[-2:]
        lines = []
        for h in recent:
            who = "用户" if h.get("role") == "user" else "Alleys"
            txt = str(h.get("content", ""))[:80]
            lines.append(f"{who}: {txt}")
        if lines:
            history_block = "最近对话：\n" + "\n".join(lines) + "\n"

    system = (
        "你是陪看智能体 Alleys 的「用户理解模块」。"
        "读用户这一句话, 判断 4 件事, 只输出严格 JSON, 不要解释, 不要 markdown.\n\n"
        "{\n"
        '  "task": "意图任务, 决定 Alleys 能看到哪些上下文",\n'
        '  "task_confidence": "对 task 的把握 0.0~1.0",\n'
        '  "emotion": "当下情绪",\n'
        '  "emotion_confidence": "对 emotion 的把握 0.0~1.0",\n'
        '  "user_state": "根据当前文本直接看出的状态或诉求"\n'
        "}\n\n"
        "【task 取值】\n"
        "- chitchat: 寒暄/情绪表达/感谢, 完全不涉及剧情\n"
        "- companion: 对剧情或角色随口反应/吐槽, 不是提问\n"
        "- deictic: 指当前画面提问 (这一幕/谁在说话/这个动作)\n"
        "- knowledge: 明确问剧情/角色/因果\n"
        "- meta: 问整集梗概\n"
        "- refuse: 剧外信息 (演员/现实/花絮) 或与剧情无关\n"
        "模糊时给 chitchat, 并把 task_confidence 降低.\n\n"
        "【emotion 取值】joy / surprised / curious / confused / sad / angry / frustrated / neutral\n"
        "含蓄、反讽、敷衍语气要结合上下文谨慎判断; 不确定就 neutral, emotion_confidence 降低.\n\n"
        "【user_state 规则 (重要, 防幻觉)】\n"
        "- 只写当前文本里能直接看出的状态或诉求, 一句陈述, ≤25 字.\n"
        "- 不要推测文本之外的动机、性格或长篇心理.\n"
        "- 不要以「用户」开头. 例: 被刘星逗乐了, 想一起笑 / 没看懂这一幕, 想要解释.\n"
        "- 拿不准就写: 无明显状态."
    )

    user = (
        f"当前在看：{video_label or '未知'}\n"
        f"播放进度：{time_str}\n"
        f"{history_block}"
        f"用户这句话：{query}\n\n"
        "输出 JSON。"
    )

    try:
        from src.core.config import get_config
        from src.core.llm.base_client import BaseLLMClient

        cfg = get_config()
        client = BaseLLMClient(model=cfg.model_intent)
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=120,
            enable_thinking=False,
        )
        # 解析 JSON (LLM 偶尔带 ```json 或多余文本; 用 extract_json_obj 处理嵌套/代码块)
        from src.core.helpers.text_utils import extract_json_obj
        obj = extract_json_obj(raw or "")
        if not obj:
            return IntentResult(
                task=_parse_task(raw),
                task_confidence=0.0,
                emotion="neutral",
                emotion_confidence=0.0,
                user_state="无明显状态",
            )

        task = _parse_task(str(obj.get("task", "")))
        if task not in _VALID_TASKS:
            task = "kb"
        emotion = _parse_emotion(str(obj.get("emotion", "")))

        def _f(key: str) -> float:
            try:
                v = float(obj.get(key, 0.0))
            except (TypeError, ValueError):
                v = 0.0
            return max(0.0, min(1.0, v))

        user_state = str(obj.get("user_state", "") or "无明显状态").strip()
        if not user_state:
            user_state = "无明显状态"

        return IntentResult(
            task=task,
            task_confidence=_f("task_confidence"),
            emotion=emotion,
            emotion_confidence=_f("emotion_confidence"),
            user_state=user_state,
        )
    except Exception as e:
        logger.warning(f"[intent_router] LLM 理解失败, 走最小上下文: {e}")
        return _FALLBACK
