"""语义意图路由 (Semantic Routing).

预定义 intent 的语义描述 (自然语言, 不是关键词).
query embedding 和 intent 描述 embedding 余弦匹配 → 最相关 intent.
纯语义理解, 不用正则/关键词. 用户没说过那些字符也能匹配到同一意图.

扩展新 intent: 在 INTENT_CATALOGUE 加描述 + companion.py 加对应处理.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 意图目录: intent_name → {descriptions, desc}
# descriptions 是"用户可能的问法"的自然语言描述 (多条, 取平均向量)
# embedding 捕获语义变体 — 用户换种说法也能匹配到同一 intent
# ============================================================

INTENT_CATALOGUE: dict[str, dict] = {
    "deictic": {
        "descriptions": [
            "用户指着当前视频画面提问, 问这一刻发生了什么",
            "这个画面里谁在说话, 这个动作是什么意思",
            "刚才那个镜头, 此刻的对白, 这一刻的表情",
            "视频里这个人是谁, 这个场景在干什么",
            "暂停在这里问画面内容, 这一句台词",
        ],
        "desc": "指代当前画面 → 只用时间邻域检索 (跳过全集向量)",
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
    "chitchat": {
        "descriptions": [
            "用户打招呼, 表达情绪, 闲聊",
            "你好, 谢谢, 辛苦了, 加油",
            "哈哈, 嘿嘿, 拜拜, 早安晚安",
            "情绪表达, 鼓励, 感谢, 问候",
        ],
        "desc": "闲聊 → 不查 KB, 直接人设回应",
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
    print(
        f"[intent_router] 预建 {len(_intent_vectors)} 个语义意图向量: "
        f"{list(_intent_vectors.keys())}",
        flush=True,
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


# ============================================================
# LLM 意图理解 (qwen-turbo, 真正的语义理解)
# ============================================================

def llm_route_intent(
    query: str,
    video_time: float | None = None,
) -> str:
    """LLM 意图理解 (qwen-turbo, 快+便宜, ~300-500ms).

    真正的语义理解: LLM 读 query + 当前视频时间, 判断意图.
    比 embedding 匹配更强 (能理解复杂/模糊/多意图, 能发现新意图).

    Returns: intent name (deictic/meta/chitchat/kb/refuse). 失败走 kb.
    """
    time_str = f"{video_time:.0f}s" if video_time is not None else "未知"
    prompt = (
        "判断用户意图, 只输出一个意图词 (不解释, 不加标点):\n"
        "- deictic: 指代当前视频画面提问 (画面里谁在说话/这个动作)\n"
        "- meta: 问整集剧情梗概 (这集讲了什么)\n"
        "- chitchat: 闲聊/打招呼/情绪\n"
        "- kb: 具体剧情/角色/事件问题\n"
        "- refuse: 超出范围 (演员现实信息/幕后)\n\n"
        f"当前视频时间: {time_str}\n"
        f"用户问题: {query}\n\n意图:"
    )
    try:
        import dashscope
        from src.core.config import get_config
        dashscope.api_key = get_config().dashscope_api_key
        # dashscope SDK 直调 (比 OpenAI SDK 快 1.7s: 0.7s vs 2.4s)
        resp = dashscope.Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            max_tokens=10,
            temperature=0,
            result_format="message",
            enable_thinking=False,
        )
        raw = ""
        if resp.status_code == 200:
            raw = resp.output.choices[0].message.content
        intent = (raw or "").strip().lower().rstrip("。.,，")
        # 容错映射 (LLM 可能输出中文或带标点)
        if any(x in intent for x in ["deictic", "指代", "画面", "当前"]):
            return "deictic"
        if any(x in intent for x in ["meta", "梗概", "整集", "概括"]):
            return "meta"
        if any(x in intent for x in ["chitchat", "闲聊", "聊天"]):
            return "chitchat"
        if any(x in intent for x in ["refuse", "拒答", "超出"]):
            return "refuse"
        return "kb"  # 默认: 剧情问题
    except Exception as e:
        print(f"[intent_router] LLM 意图理解失败, 走默认 kb: {e}", flush=True)
        return "kb"
