"""L1 用户画像 + L2 作品画像增量更新 (会话级, 慢节奏).

触发时机: companion 在非 refuse 任务后累加计数, 达到阈值 (PROFILE_UPDATE_THRESHOLD)
         才触发一次. 不每轮写, 避免 LLM 调用爆炸 + 画像抖动.

实现: qwen3.7-flash 读最近 N 轮对话 + 旧画像, 输出 JSON, UPSERT.
"""

from __future__ import annotations

import json

from src.agent.profile_store import (
    load_user_profile,
    save_profile,
    load_show_profile,
    save_show_profile,
)
from src.core.logging import get_logger

logger = get_logger()


_VALID_STYLE = {"吐槽型", "分析型", "陪伴型", "提问型", "混合"}
_VALID_SPOILER = {"接受", "谨慎", "拒绝"}
_VALID_HUMOR = {"高", "中", "低"}
_VALID_MOTIVATION = {"推理探索型", "情绪共鸣型", "角色陪伴型", "剧情消费型"}
_VALID_SENTIMENT = {"positive", "neutral", "negative"}


def _pick(value: str | None, allowed: set[str]) -> str | None:
    if not value:
        return None
    v = str(value).strip()
    return v if v in allowed else None


def _render_history(chat_history: list[dict]) -> tuple[str, int]:
    """渲染最近 N 轮对话为文本, 返回 (text, n_rounds)."""
    recent = chat_history[-20:]
    lines = []
    for h in recent:
        who = "用户" if h.get("role") == "user" else "Alleys"
        txt = str(h.get("content", ""))[:120]
        lines.append(f"{who}: {txt}")
    return "\n".join(lines), len(recent)


# ============================================================
# L1: 用户画像
# ============================================================

def maybe_update_user_profile(user_id: str, chat_history: list[dict]) -> None:
    """读旧画像 + 最近对话 → qwen3.7-flash → UPSERT. 失败静默 (后台任务)."""
    if not chat_history or len(chat_history) < 4:
        return

    old = load_user_profile(user_id) or {}
    history_text, n_recent = _render_history(chat_history)

    old_brief = (
        f"interaction_style={old.get('interaction_style') or '未知'}, "
        f"spoiler_tolerance={old.get('spoiler_tolerance') or '未知'}, "
        f"humor_level={old.get('humor_level') or '未知'}, "
        f"engagement_motivation={old.get('engagement_motivation') or '未知'}, "
        f"confidence={old.get('confidence', 0):.2f}"
    )

    system = (
        "你在更新陪看智能体对某个用户的长期画像。只输出严格 JSON，不要解释、不要 markdown。\n\n"
        "{\n"
        '  "interaction_style": "吐槽型/分析型/陪伴型/提问型/混合",\n'
        '  "spoiler_tolerance": "接受/谨慎/拒绝",\n'
        '  "humor_level": "高/中/低",\n'
        '  "engagement_motivation": "推理探索型/情绪共鸣型/角色陪伴型/剧情消费型",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "字段说明:\n"
        "- interaction_style:\n"
        "  · 吐槽型: 喜欢吐槽、评价剧情\n"
        "  · 分析型: 喜欢分析人物、剧情逻辑、主题\n"
        "  · 陪伴型: 关注情绪交流、一起看剧的感觉\n"
        "  · 提问型: 主要通过问题推进观看\n"
        "  · 混合: 多种明显存在\n"
        "- spoiler_tolerance: 用户接受剧情信息程度\n"
        "- humor_level: 用户是否喜欢玩笑、接梗、轻松互动\n"
        "- engagement_motivation: 用户观看和讨论的主要动力\n"
        "  · 推理探索型: 喜欢猜测、分析未知\n"
        "  · 情绪共鸣型: 关注情感体验\n"
        "  · 角色陪伴型: 特别关注角色命运\n"
        "  · 剧情消费型: 主要关注事件发展\n"
        "- confidence:\n"
        "  · 0.9-1.0: 明确表达且多次验证\n"
        "  · 0.6-0.8: 有明显行为依据\n"
        "  · 0.3-0.5: 弱推断\n"
        "  · 0-0.2: 基本无法判断\n\n"
        "不要 invent 字段。\n\n"
        "【关键】对话中「用户:」开头的才是用户本人的发言;\n"
        "「Alleys:」开头的是智能体的回复, 仅作上下文理解用途, 绝不能当成用户的风格或态度.\n"
        "所有画像判断必须基于「用户:」行的内容, 不要被 Alleys 的语气/观点带偏."
    )
    user = (
        f"当前画像: {old_brief}\n\n"
        f"最近对话:\n{history_text}\n\n"
        "输出 JSON。"
    )

    try:
        from src.core.llm.base_client import BaseLLMClient
        client = BaseLLMClient(model="qwen3.7-flash")
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=150,
            enable_thinking=False,
        )
        from src.core.helpers.text_utils import extract_json_obj
        obj = extract_json_obj(raw or "")
        if not obj:
            return

        # 合并: 新值优先, 新值非法时保留旧值
        style = _pick(obj.get("interaction_style"), _VALID_STYLE) or old.get("interaction_style")
        spoiler = _pick(obj.get("spoiler_tolerance"), _VALID_SPOILER) or old.get("spoiler_tolerance")
        humor = _pick(obj.get("humor_level"), _VALID_HUMOR) or old.get("humor_level")
        motivation = _pick(obj.get("engagement_motivation"), _VALID_MOTIVATION) or old.get("engagement_motivation")
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        conf = min(conf, n_recent / 20.0)

        save_profile(
            user_id,
            interaction_style=style,
            spoiler_tolerance=spoiler,
            humor_level=humor,
            engagement_motivation=motivation,
            confidence=conf,
        )
        logger.info(
            f"[profile] 更新 user={user_id[:8]} style={style} spoiler={spoiler} "
            f"humor={humor} motivation={motivation} conf={conf:.2f}"
        )
    except Exception as e:
        logger.warning(f"[profile] 更新失败 (非致命): {e}")


# ============================================================
# L2: 作品画像
# ============================================================

def maybe_update_show_profile(
    user_id: str, show: str, chat_history: list[dict],
) -> None:
    """L2 作品画像增量更新: 角色偏好 + 剧情兴趣 + 评价倾向."""
    if not show or not chat_history or len(chat_history) < 4:
        return

    old = load_show_profile(user_id, show) or {}
    old_fav = old.get("favorite_characters") or []
    old_att = old.get("attention_characters") or []
    old_ops = old.get("character_opinions") or []
    old_themes = old.get("theme_preferences") or []
    old_disliked = old.get("disliked_elements") or []

    history_text, n_recent = _render_history(chat_history)

    system = (
        "你在更新陪看智能体对某个用户在某部作品中的长期记忆。\n\n"
        "只输出严格 JSON，不要解释，不要 markdown。\n\n"
        "目标:\n"
        "记录用户在当前作品中的角色偏好、剧情兴趣和评价倾向，用于后续陪看互动。\n\n"
        "输出:\n\n"
        "{\n"
        ' "favorite_characters":[],\n'
        ' "attention_characters":[],\n'
        ' "character_opinions":[],\n'
        ' "theme_preferences":[],\n'
        ' "disliked_elements":[],\n'
        ' "confidence":0.0\n'
        "}\n\n"
        "规则:\n\n"
        "1. 只根据「用户:」内容判断。\n"
        "2. 「Alleys:」观点不能作为用户评价。\n"
        "3. 用户明确表达喜欢/讨厌 > 长期行为信号 > 单次关注。\n"
        "4. 不确定时保留旧值，不新增猜测。\n\n"
        "字段:\n\n"
        "favorite_characters:\n"
        "用户明显喜欢、认可、心疼的角色。\n\n"
        "attention_characters:\n"
        "用户关注、频繁询问、讨论较多的角色。\n"
        "注意:\n"
        "关注不等于喜欢。\n\n"
        "character_opinions:\n"
        "格式:\n"
        "{\n"
        '"character":"角色名",\n'
        '"opinion":"一句评价",\n'
        '"sentiment":"positive/neutral/negative"\n'
        "}\n\n"
        "theme_preferences:\n"
        "用户喜欢的剧情主题，例如:\n"
        "权谋、爱情、成长、悬疑、人性等。\n\n"
        "disliked_elements:\n"
        "用户明确不喜欢的内容，例如:\n"
        "强行反转、拖沓剧情、某类角色。\n\n"
        "confidence:\n"
        "0-1:\n"
        "0.9-1.0 明确多次表达\n"
        "0.6-0.8 行为明显\n"
        "0.3-0.5 弱推断\n"
        "0-0.2 不确定\n\n"
        "不要编造角色。\n"
        "不要 invent 字段。"
    )
    user = (
        f"当前剧: {show}\n"
        f"已记下的喜好角色: {old_fav}\n"
        f"已记下的关注角色: {old_att}\n"
        f"已记下的评价: {json.dumps(old_ops, ensure_ascii=False)}\n"
        f"已记下的主题偏好: {old_themes}\n"
        f"已记下的不喜欢: {old_disliked}\n\n"
        f"最近对话:\n{history_text}\n\n"
        "输出 JSON。"
    )

    try:
        from src.core.llm.base_client import BaseLLMClient
        client = BaseLLMClient(model="qwen3.7-flash")
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=300,
            enable_thinking=False,
        )
        from src.core.helpers.text_utils import extract_json_obj
        obj = extract_json_obj(raw or "")
        if not obj:
            return

        # favorite_characters: 合并去重, 保留最多 8 个
        new_fav = [c for c in (obj.get("favorite_characters") or []) if isinstance(c, str) and c.strip()]
        fav = list(dict.fromkeys([c.strip() for c in (old_fav + new_fav) if c.strip()]))[:8]

        # attention_characters: 合并去重, 保留最多 8 个
        new_att = [c for c in (obj.get("attention_characters") or []) if isinstance(c, str) and c.strip()]
        att = list(dict.fromkeys([c.strip() for c in (old_att + new_att) if c.strip()]))[:8]

        # character_opinions: 合并 (同角色新评价覆盖旧评价)
        merged: dict[str, dict] = {o.get("character"): o for o in old_ops if isinstance(o, dict)}
        for o in (obj.get("character_opinions") or []):
            if not isinstance(o, dict):
                continue
            ch = (o.get("character") or "").strip()
            if not ch:
                continue
            sentiment = o.get("sentiment") if o.get("sentiment") in _VALID_SENTIMENT else "neutral"
            opinion = str(o.get("opinion", "")).strip()[:40]
            merged[ch] = {"character": ch, "opinion": opinion, "sentiment": sentiment}
        ops = list(merged.values())[:10]

        # theme_preferences: 合并去重
        new_themes = [t for t in (obj.get("theme_preferences") or []) if isinstance(t, str) and t.strip()]
        themes = list(dict.fromkeys([t.strip() for t in (old_themes + new_themes) if t.strip()]))[:8]

        # disliked_elements: 合并去重
        new_disliked = [d for d in (obj.get("disliked_elements") or []) if isinstance(d, str) and d.strip()]
        disliked = list(dict.fromkeys([d.strip() for d in (old_disliked + new_disliked) if d.strip()]))[:8]

        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        conf = min(conf, n_recent / 20.0)

        save_show_profile(
            user_id, show,
            favorite_characters=fav,
            attention_characters=att,
            character_opinions=ops,
            theme_preferences=themes,
            disliked_elements=disliked,
            confidence=conf,
        )
        logger.info(
            f"[show_profile] 更新 user={user_id[:8]} show={show} "
            f"fav={len(fav)} att={len(att)} ops={len(ops)} "
            f"themes={len(themes)} disliked={len(disliked)} conf={conf:.2f}"
        )
    except Exception as e:
        logger.warning(f"[show_profile] 更新失败 (非致命): {e}")
