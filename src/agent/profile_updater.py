"""L1 用户画像 + L2 作品画像增量更新.

双轨采集:
  轻量轨道 (maybe_update_user_profile_instant): 每轮调用 qwen-turbo,
    只提取表现层两字段 (alleys_attitude / alleys_response_preference),
    直接覆盖, 无置信度, 保证用户当下指令即时生效.
  完整轨道 (maybe_update_user_profile): 每 PROFILE_UPDATE_THRESHOLD 轮调用
    qwen3.7-flash, 提取全部内核层字段, conf_stable 用 EWMA 维护 +
    字段门控 (新信号置信度足够才接受新字段值), 防止噪声污染稳定画像.

实现: 读最近 N 轮对话 + 旧画像, 输出 JSON, UPSERT.
"""

from __future__ import annotations

import json

from src.agent.profile_store import (
    load_user_profile,
    save_profile,
    update_instant_fields,
    load_show_profile,
    save_show_profile,
)
from src.core.logging import get_logger

logger = get_logger()

# 完整轨道 EWMA 融合系数 (旧画像权重高, 内核层慢更新)
_EWMA_ALPHA = 0.3
# 字段门控: 新信号置信度达到该值才接受新字段值
_FIELD_GATE = 0.5


_VALID_STYLE = {"吐槽型", "分析型", "陪伴型", "提问型", "混合"}
_VALID_SPOILER = {"接受", "谨慎", "拒绝"}
_VALID_HUMOR = {"高", "中", "低"}
_VALID_MOTIVATION = {"推理探索型", "情绪共鸣型", "角色陪伴型", "剧情消费型"}
_VALID_INITIATIVE = {"主动型", "被动型", "混合"}
_VALID_RESPONSE_PREF = {"反问引导", "直接表态", "拱火加码", "冷静降温"}
_VALID_TEASING = {"能被吐槽", "只能吐槽剧情", "完全不能"}
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


def _parse_conf_stable(obj: dict, n_recent: int) -> float:
    """LLM conf_stable 取整 + 按样本量阻尼 (样本越少越保守)."""
    try:
        conf = float(obj.get("conf_stable", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return min(max(0.0, min(1.0, conf)), n_recent / 20.0)


def _parse_confidence(obj: dict, n_recent: int) -> float:
    """L2 专用: LLM confidence 取整 + 按样本量阻尼 (L2 保留整表 confidence)."""
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return min(max(0.0, min(1.0, conf)), n_recent / 20.0)


def _ewma(new_signal: float, old: float, alpha: float = _EWMA_ALPHA) -> float:
    """指数加权移动平均: 旧值惯性大, 新信号持续推才改方向."""
    return round(alpha * new_signal + (1.0 - alpha) * old, 3)


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


def _call_profile_llm(system: str, user: str, max_tokens: int, model: str = "qwen3.7-flash") -> dict | None:
    """读旧画像+对话, 返回解析后的 JSON dict.

    model: 轻量轨道用 qwen-turbo (便宜, 每轮跑), 完整轨道用 qwen3.7-flash.
    """
    from src.core.llm.base_client import BaseLLMClient
    from src.core.helpers.text_utils import extract_json_obj
    raw = BaseLLMClient(model=model).chat(
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


_L1_INSTANT_SYSTEM_FALLBACK = (
    "你在更新陪看智能体对某个用户的当下印象。只输出严格 JSON，不要解释、不要 markdown。\n"
    '{"alleys_attitude":"","alleys_response_preference":""}\n'
    "alleys_attitude: 用户**当下**对 Alleys 的直接态度/指令 (抱怨/表扬/要求), "
    "格式=态度+行为指令 (如 '嫌啰嗦,每条≤30字'). 无明显信号填空字符串.\n"
    "alleys_response_preference: 用户当下希望的回应策略 (反问引导/直接表态/拱火加码/冷静降温). 无信号填空.\n"
    "只根据「用户:」直接对 Alleys 说的话判断, 剧情相关内容不写进这两字段.\n"
    "这两字段每轮直接覆盖旧值 (表现层易变, 用户当下说了什么就是什么)."
)

_L1_SYSTEM_FALLBACK = (
    "你在更新陪看智能体对某个用户的长期画像。只输出严格 JSON，不要解释、不要 markdown。\n"
    '{"interaction_style":"吐槽型/分析型/陪伴型/提问型/混合",'
    '"interaction_initiative":"主动型/被动型/混合",'
    '"engagement_motivation":"推理探索型/情绪共鸣型/角色陪伴型/剧情消费型",'
    '"humor_level":"高/中/低","teasing_tolerance":"能被吐槽/只能吐槽剧情/完全不能",'
    '"spoiler_tolerance":"接受/谨慎/拒绝",'
    '"pet_peeves":[],"alleys_attitude":"","alleys_response_preference":"","conf_stable":0.0}\n'
    "L1 内核层决定 Alleys 怎么回, 字段值要可执行. 弱信号字段填空字符串或空数组.\n"
    "alleys_attitude / alleys_response_preference 是表现层, 由轻量轨道单独更新, 这里填旧值或空即可.\n"
    "conf_stable: 你对上述内核层字段的整体置信度. "
    "0.9-1.0 明确多次表达 / 0.6-0.8 行为明显 / 0.3-0.5 弱推断 / 0-0.2 不确定.\n"
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

def maybe_update_user_profile_instant(user_id: str, chat_history: list[dict]) -> None:
    """轻量轨道 (每轮): 只更新表现层两字段, qwen-turbo, 直接覆盖.

    表现层易变 (用户当下指令), 不用 EWMA 不用门控, 说了什么就是什么.
    失败静默 (后台任务).
    """
    if not chat_history or not user_id:
        return
    history_text, _ = _render_history(chat_history)

    system = _get_profile_prompt("l1_instant_system", _L1_INSTANT_SYSTEM_FALLBACK)
    user = f"最近对话:\n{history_text}\n\n输出 JSON。"

    try:
        obj = _call_profile_llm(system, user, max_tokens=100, model="qwen-turbo")
        if not obj:
            return
        attitude_raw = str(obj.get("alleys_attitude", "") or "").strip()
        attitude = attitude_raw[:50] if attitude_raw and attitude_raw != "未知" else None
        response_pref = _pick(obj.get("alleys_response_preference"), _VALID_RESPONSE_PREF)
        if attitude is None and response_pref is None:
            return
        update_instant_fields(user_id, attitude, response_pref)
        logger.info(
            f"[profile][instant] user={user_id[:8]} attitude={attitude} "
            f"pref={response_pref}"
        )
    except Exception as e:
        logger.warning(f"[profile][instant] 更新失败 (非致命): {e}")


def maybe_update_user_profile(user_id: str, chat_history: list[dict]) -> None:
    """完整轨道 (每 PROFILE_UPDATE_THRESHOLD 轮): 更新内核层全部字段.

    策略:
      - conf_stable 用 EWMA 维护 (旧画像惯性大, 新信号持续推才改方向)
      - 字段门控: 新信号置信度 >= _FIELD_GATE 才接受新字段值, 否则保留旧值
      - 表现层字段由轻量轨道负责, 此处随完整轨道带回旧值即可
    失败静默 (后台任务).
    """
    if not chat_history or len(chat_history) < 2:
        return

    old = load_user_profile(user_id) or {}
    history_text, n_recent = _render_history(chat_history)
    old_conf = float(old.get("conf_stable", 0) or 0)

    old_brief = (
        f"interaction_style={old.get('interaction_style') or '未知'}, "
        f"interaction_initiative={old.get('interaction_initiative') or '未知'}, "
        f"engagement_motivation={old.get('engagement_motivation') or '未知'}, "
        f"humor_level={old.get('humor_level') or '未知'}, "
        f"teasing_tolerance={old.get('teasing_tolerance') or '未知'}, "
        f"spoiler_tolerance={old.get('spoiler_tolerance') or '未知'}, "
        f"pet_peeves={old.get('pet_peeves') or []}, "
        f"conf_stable={old_conf:.2f}"
    )

    system = _get_profile_prompt("l1_system", _L1_SYSTEM_FALLBACK)
    user = (
        f"当前画像: {old_brief}\n\n"
        f"最近对话:\n{history_text}\n\n"
        "输出 JSON。"
    )

    try:
        obj = _call_profile_llm(system, user, max_tokens=220)
        if not obj:
            return

        new_signal = _parse_conf_stable(obj, n_recent)
        gated = new_signal >= _FIELD_GATE  # 信号够强才接受新字段值

        # 字段门控: 通过才取新值, 否则保留旧值 (防低置信度污染稳定画像)
        style = _pick(obj.get("interaction_style"), _VALID_STYLE) if gated else None
        spoiler = _pick(obj.get("spoiler_tolerance"), _VALID_SPOILER) if gated else None
        humor = _pick(obj.get("humor_level"), _VALID_HUMOR) if gated else None
        motivation = _pick(obj.get("engagement_motivation"), _VALID_MOTIVATION) if gated else None
        initiative = _pick(obj.get("interaction_initiative"), _VALID_INITIATIVE) if gated else None
        teasing = _pick(obj.get("teasing_tolerance"), _VALID_TEASING) if gated else None
        # pet_peeves: list 去重限长, 新值非空且过门控才采用
        peeves: list[str] = []
        if gated:
            raw_peeves = obj.get("pet_peeves") or []
            if isinstance(raw_peeves, list):
                seen: set[str] = set()
                for p in raw_peeves:
                    if isinstance(p, str):
                        p = p.strip()
                        if p and p not in seen:
                            seen.add(p)
                            peeves.append(p)
                peeves = peeves[:6]

        # 保留兜底: 新值非法或未过门控 → 保留旧值
        style = style or old.get("interaction_style")
        spoiler = spoiler or old.get("spoiler_tolerance")
        humor = humor or old.get("humor_level")
        motivation = motivation or old.get("engagement_motivation")
        initiative = initiative or old.get("interaction_initiative")
        teasing = teasing or old.get("teasing_tolerance")
        peeves = peeves or old.get("pet_peeves") or []

        # 表现层: 完整轨道带上当前值 (避免覆盖轻量轨道刚写的最新态度)
        attitude = old.get("alleys_attitude")
        response_pref = old.get("alleys_response_preference")

        # conf_stable EWMA: 旧值惯性大, 持续同向信号才拉得动
        conf_stable = _ewma(new_signal, old_conf)

        save_profile(
            user_id,
            interaction_style=style,
            spoiler_tolerance=spoiler,
            humor_level=humor,
            engagement_motivation=motivation,
            interaction_initiative=initiative,
            teasing_tolerance=teasing,
            pet_peeves=peeves,
            alleys_attitude=attitude,
            alleys_response_preference=response_pref,
            conf_stable=conf_stable,
        )
        logger.info(
            f"[profile] 更新 user={user_id[:8]} style={style} spoiler={spoiler} "
            f"humor={humor} motivation={motivation} init={initiative} "
            f"teasing={teasing} peeves={peeves} "
            f"signal={new_signal:.2f} gated={gated} conf_stable={conf_stable:.2f}"
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
