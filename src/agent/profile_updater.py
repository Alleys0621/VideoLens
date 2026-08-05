"""L1 用户画像 + L2 作品画像增量更新 (会话级, 慢节奏).

触发时机: companion 累加计数, 达到阈值 (PROFILE_UPDATE_THRESHOLD)
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


def _parse_confidence(obj: dict, n_recent: int) -> float:
    """LLM confidence 取整 + 按样本量阻尼."""
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return min(max(0.0, min(1.0, conf)), n_recent / 20.0)


def _load_main_cast(show: str) -> list[str]:
    """从 _global/character_profiles.json 读主角名单, 用于 prompt 提示 LLM 归一角色名.

    防"夏雪/小雪"这种昵称/全名分裂. 没文件 / 无主角 → 返回 [], 调用方跳过 cast 提示.
    """
    import os
    try:
        from src.core.config import get_config
        from src.core.helpers.json_utils import load_json
        cfg = get_config()
        p = os.path.join(cfg.output_root, "_global", "character_profiles.json")
        if not os.path.isfile(p):
            return []
        data = load_json(p)
        profiles = data.get("profiles") or []
        names = [prof.get("name") for prof in profiles if prof.get("role") == "主角" and prof.get("name")]
        return names[:8]
    except Exception as e:
        logger.debug(f"_load_main_cast 失败 (非致命): {e}")
        return []


def _call_profile_llm(system: str, user: str, max_tokens: int) -> dict | None:
    """qwen3.7-flash 读旧画像+对话, 返回解析后的 JSON dict."""
    from src.core.llm.base_client import BaseLLMClient
    from src.core.helpers.text_utils import extract_json_obj
    raw = BaseLLMClient(model="qwen3.7-flash").chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
        enable_thinking=False,
    )
    return extract_json_obj(raw or "")


# ============================================================
# Prompt 读取 (yaml profile_prompts, 缺配置时用代码内 fallback)
# ============================================================

def _get_profile_prompt(key: str, fallback: str) -> str:
    """从 yaml `profile_prompts.{key}` 读 prompt; 未配置用 fallback."""
    try:
        from src.core.config import get_config
        pp = get_config().prompts.get("profile_prompts", {}) or {}
        val = (pp.get(key) or "").strip()
        return val if val else fallback
    except Exception:
        return fallback


_L1_SYSTEM_FALLBACK = (
    "你在更新陪看智能体对某个用户的长期画像。只输出严格 JSON，不要解释、不要 markdown。\n"
    '{"interaction_style":"吐槽型/分析型/陪伴型/提问型/混合",'
    '"spoiler_tolerance":"接受/谨慎/拒绝","humor_level":"高/中/低",'
    '"engagement_motivation":"推理探索型/情绪共鸣型/角色陪伴型/剧情消费型","confidence":0.0}\n'
    "confidence: 0.9-1.0 明确多次表达 / 0.6-0.8 行为明显 / 0.3-0.5 弱推断 / 0-0.2 不确定.\n"
    "【关键】只有「用户:」开头的是用户本人发言, 「Alleys:」是智能体回复, 不能当用户风格.\n"
    "覆盖规则: 最近对话与旧值冲突时以最近为准, 允许覆盖旧值. "
    "用户抱怨 AI 回答质量(别瞎编/答错了)不是剧情偏好, 只作 interaction_style 判断参考."
)

_L2_SYSTEM_FALLBACK = (
    "你在更新陪看智能体对某个用户在某部作品中的长期记忆。只输出严格 JSON，不要解释、不要 markdown。\n"
    '{"favorite_characters":[],"attention_characters":[],"character_opinions":[],'
    '"theme_preferences":[],"disliked_elements":[],"confidence":0.0}\n'
    "只根据「用户:」内容判断,「Alleys:」不能作为用户评价. "
    "明确喜欢/讨厌 > 长期行为 > 单次关注. 不要编造角色.\n"
    "覆盖规则: 最近对话与旧值冲突时以最近为准, 允许修改/移除旧 favorite/opinion/disliked. "
    "角色字段只收剧中角色名, 演员真名(如张一山)不进. 用户抱怨 AI 回答质量不进本画像. "
    "互斥: 同一角色不能同时进 favorite 和 disliked, 冲突以 opinion 的 sentiment 为准."
)


# ============================================================
# L1: 用户画像
# ============================================================

def maybe_update_user_profile(user_id: str, chat_history: list[dict]) -> None:
    """读旧画像 + 最近对话 → qwen3.7-flash → UPSERT. 失败静默 (后台任务)."""
    if not chat_history or len(chat_history) < 2:
        return

    old = load_user_profile(user_id) or {}
    history_text, n_recent = _render_history(chat_history)

    old_brief = (
        f"interaction_style={old.get('interaction_style') or '未知'}, "
        f"spoiler_tolerance={old.get('spoiler_tolerance') or '未知'}, "
        f"humor_level={old.get('humor_level') or '未知'}, "
        f"engagement_motivation={old.get('engagement_motivation') or '未知'}, "
        f"alleys_attitude={old.get('alleys_attitude') or '未知'}, "
        f"confidence={old.get('confidence', 0):.2f}"
    )

    system = _get_profile_prompt("l1_system", _L1_SYSTEM_FALLBACK)
    user = (
        f"当前画像: {old_brief}\n\n"
        f"最近对话:\n{history_text}\n\n"
        "输出 JSON。"
    )

    try:
        obj = _call_profile_llm(system, user, max_tokens=150)
        if not obj:
            return

        # 合并: 新值优先, 新值非法时保留旧值
        style = _pick(obj.get("interaction_style"), _VALID_STYLE) or old.get("interaction_style")
        spoiler = _pick(obj.get("spoiler_tolerance"), _VALID_SPOILER) or old.get("spoiler_tolerance")
        humor = _pick(obj.get("humor_level"), _VALID_HUMOR) or old.get("humor_level")
        motivation = _pick(obj.get("engagement_motivation"), _VALID_MOTIVATION) or old.get("engagement_motivation")
        # alleys_attitude: 自由文本, 保留最近的有效值
        attitude_raw = str(obj.get("alleys_attitude", "") or "").strip()
        if attitude_raw and attitude_raw != "未知":
            attitude = attitude_raw[:50]
        else:
            attitude = old.get("alleys_attitude")
        conf = _parse_confidence(obj, n_recent)

        save_profile(
            user_id,
            interaction_style=style,
            spoiler_tolerance=spoiler,
            humor_level=humor,
            engagement_motivation=motivation,
            alleys_attitude=attitude,
            confidence=conf,
        )
        logger.info(
            f"[profile] 更新 user={user_id[:8]} style={style} spoiler={spoiler} "
            f"humor={humor} motivation={motivation} attitude={attitude} conf={conf:.2f}"
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
    if not show or not chat_history or len(chat_history) < 2:
        return

    old = load_show_profile(user_id, show) or {}
    old_fav = old.get("favorite_characters") or []
    old_att = old.get("attention_characters") or []
    old_ops = old.get("character_opinions") or []
    old_themes = old.get("theme_preferences") or []
    old_disliked = old.get("disliked_elements") or []

    history_text, n_recent = _render_history(chat_history)

    system = _get_profile_prompt("l2_system", _L2_SYSTEM_FALLBACK)
    cast = _load_main_cast(show)
    cast_hint = f"本剧主角团 (角色名必须归一到这些全名): {', '.join(cast)}\n" if cast else ""
    user = (
        f"当前剧: {show}\n"
        + cast_hint +
        f"已记下的喜好角色: {old_fav}\n"
        f"已记下的关注角色: {old_att}\n"
        f"已记下的评价: {json.dumps(old_ops, ensure_ascii=False)}\n"
        f"已记下的主题偏好: {old_themes}\n"
        f"已记下的不喜欢: {old_disliked}\n\n"
        f"最近对话:\n{history_text}\n\n"
        "输出 JSON。"
    )

    try:
        obj = _call_profile_llm(system, user, max_tokens=300)
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

        conf = _parse_confidence(obj, n_recent)

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
