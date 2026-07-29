"""L1 用户画像增量更新 (会话级, 慢节奏).

触发时机: companion 在非 refuse 任务后累加计数, 达到阈值 (PROFILE_UPDATE_THRESHOLD)
         才触发一次. 不每轮写, 避免 LLM 调用爆炸 + 画像抖动.

实现: qwen-plus 读最近 N 轮对话 + 旧画像, 输出 JSON, UPSERT.
比 qwen-max 便宜, 画像更新不强求最强推理 (属于后台维护任务).
"""

from __future__ import annotations

import json
import re

from src.agent.profile_store import (
    load_user_profile,
    save_profile,
    load_show_profile,
    save_show_profile,
)


_VALID_STYLE = {"吐槽型", "分析型", "陪伴型", "提问型", "混合"}
_VALID_SPOILER = {"接受", "谨慎", "拒绝"}
_VALID_HUMOR = {"高", "中", "低"}
_VALID_SENTIMENT = {"positive", "neutral", "negative"}


def _pick(value: str | None, allowed: set[str]) -> str | None:
    if not value:
        return None
    v = str(value).strip()
    return v if v in allowed else None


def maybe_update_user_profile(user_id: str, chat_history: list[dict]) -> None:
    """读旧画像 + 最近对话 → qwen-plus → UPSERT. 失败静默 (后台任务)."""
    if not chat_history or len(chat_history) < 4:
        # 样本太少, 不更新
        return

    old = load_user_profile(user_id) or {}
    recent = chat_history[-20:]
    lines = []
    for h in recent:
        who = "用户" if h.get("role") == "user" else "Alleys"
        txt = str(h.get("content", ""))[:120]
        lines.append(f"{who}: {txt}")
    history_text = "\n".join(lines)

    old_brief = (
        f"interaction_style={old.get('interaction_style') or '未知'}, "
        f"spoiler_tolerance={old.get('spoiler_tolerance') or '未知'}, "
        f"humor_level={old.get('humor_level') or '未知'}, "
        f"confidence={old.get('confidence', 0):.2f}"
    )

    system = (
        "你在更新陪看智能体对某个用户的长期画像。只输出严格 JSON，不要解释、不要 markdown。\n\n"
        "{\n"
        '  "interaction_style": "吐槽型/分析型/陪伴型/提问型/混合",\n'
        '  "spoiler_tolerance": "接受/谨慎/拒绝",\n'
        '  "humor_level": "高/中/低",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "判定要点:\n"
        "- interaction_style: 这个人怎么跟 Alleys 聊 — 爱吐槽 / 爱分析人物动机 / 求陪伴共情 / 主要问剧情.\n"
        "- spoiler_tolerance: 是否接受被剧透后续.\n"
        "- humor_level: 接梗/玩笑的浓度.\n"
        "- confidence: 样本不足或看不准就给低分 (0.0~1.0).\n"
        "- 没把握的字段保留旧值, 不要凭空猜.\n"
        "- 不要 invent 字段."
    )
    user = (
        f"当前画像: {old_brief}\n\n"
        f"最近对话:\n{history_text}\n\n"
        "输出 JSON。"
    )

    try:
        from src.core.llm.base_client import BaseLLMClient
        client = BaseLLMClient(model="qwen-plus")
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=120,
            enable_thinking=False,
        )
        m = re.search(r"\{[^{}]*\}", raw or "")
        if not m:
            return
        obj = json.loads(m.group(0))

        # 合并: 新值优先, 新值非法时保留旧值
        style = _pick(obj.get("interaction_style"), _VALID_STYLE) or old.get("interaction_style")
        spoiler = _pick(obj.get("spoiler_tolerance"), _VALID_SPOILER) or old.get("spoiler_tolerance")
        humor = _pick(obj.get("humor_level"), _VALID_HUMOR) or old.get("humor_level")
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        # 样本数兜底: 对话少就不让 confidence 太高
        conf = min(conf, len(recent) / 20.0)

        save_profile(
            user_id,
            interaction_style=style,
            spoiler_tolerance=spoiler,
            humor_level=humor,
            confidence=conf,
        )
        print(
            f"[profile] 更新 user={user_id[:8]} style={style} spoiler={spoiler} "
            f"humor={humor} conf={conf:.2f}",
            flush=True,
        )
    except Exception as e:
        print(f"[profile] 更新失败 (非致命): {e}", flush=True)


def maybe_update_show_profile(
    user_id: str, show: str, chat_history: list[dict],
) -> None:
    """L2 作品画像增量更新: 抽取用户在这部剧里的角色喜好/评价.

    事实类记忆, 给 companion/knowledge 作 context 段. 和 L1 同节奏触发.
    """
    if not show or not chat_history or len(chat_history) < 4:
        return

    old = load_show_profile(user_id, show) or {}
    old_fav = old.get("favorite_characters") or []
    old_ops = old.get("character_opinions") or []

    recent = chat_history[-20:]
    lines = []
    for h in recent:
        who = "用户" if h.get("role") == "user" else "Alleys"
        txt = str(h.get("content", ""))[:120]
        lines.append(f"{who}: {txt}")
    history_text = "\n".join(lines)

    system = (
        "你在更新陪看智能体对某个用户「在某部剧里」的记忆。只输出严格 JSON，不要解释、不要 markdown。\n\n"
        "{\n"
        '  "favorite_characters": ["角色名"],\n'
        '  "character_opinions": [{"character":"角色名","opinion":"一句评价","sentiment":"positive/neutral/negative"}],\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "判定要点:\n"
        "- favorite_characters: 用户明显喜欢/关注/心疼的角色, 用剧中真名.\n"
        "- character_opinions: 用户对角色表达过态度的才写, 一句客观转述 (≤20字), 没表达过就不写.\n"
        "- sentiment: positive/neutral/negative.\n"
        "- confidence: 样本不足或看不准就给低分 (0.0~1.0).\n"
        "- 不要 invent 没出现过的角色; 不要推测."
    )
    user = (
        f"当前剧: {show}\n"
        f"已记下的喜好角色: {old_fav}\n"
        f"已记下的评价: {json.dumps(old_ops, ensure_ascii=False)}\n\n"
        f"最近对话:\n{history_text}\n\n"
        "输出 JSON。"
    )

    try:
        from src.core.llm.base_client import BaseLLMClient
        client = BaseLLMClient(model="qwen-plus")
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=200,
            enable_thinking=False,
        )
        m = re.search(r"\{.*\}", raw or "", flags=re.S)
        if not m:
            return
        obj = json.loads(m.group(0))

        # favorite_characters: 合并去重, 保留最多 8 个
        new_fav = [c for c in (obj.get("favorite_characters") or []) if isinstance(c, str) and c.strip()]
        fav = list(dict.fromkeys([c.strip() for c in (old_fav + new_fav) if c.strip()]))[:8]

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

        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        conf = min(conf, len(recent) / 20.0)

        save_show_profile(
            user_id, show,
            favorite_characters=fav,
            character_opinions=ops,
            confidence=conf,
        )
        print(
            f"[show_profile] 更新 user={user_id[:8]} show={show} "
            f"fav={fav} ops={len(ops)} conf={conf:.2f}",
            flush=True,
        )
    except Exception as e:
        print(f"[show_profile] 更新失败 (非致命): {e}", flush=True)
