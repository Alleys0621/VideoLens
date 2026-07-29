"""L1 用户长期画像读写 (Postgres).

存储结构化、可注入、可审计的人设级偏好 (聊法 / 剧透接受度 / 接梗浓度).
字段值用中文, DBeaver 直接可读.

职责边界:
  - 本表: "这个人什么样" (style 类) → 作为 system overlay 注入.
  - Mem0: 零散事实回忆 (喜欢刘星 / 上次聊到 XX) → 作为 context 检索.
  两者不重叠.

不每轮写: 由 companion 在非 refuse 任务后累加 messages_since_update,
达到 PROFILE_UPDATE_THRESHOLD 才用一次便宜 LLM 增量更新 (见 profile_updater).
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg.types.json import Json


PROFILE_UPDATE_THRESHOLD = 5  # 每累计 5 条非 refuse 对话触发一次增量更新
PROFILE_INJECT_MIN_CONFIDENCE = 0.5  # confidence 低于此值不注入 system overlay


def _conninfo() -> str:
    from src.core.config import get_config
    return get_config().postgres_url


def _is_uuid(v: str | None) -> bool:
    """轻量校验, 避免非 uuid user_id (如 'default'/'smoke') 触发 FK 报错刷屏."""
    if not v or not isinstance(v, str):
        return False
    import re as _re
    return bool(_re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", v))


def _connect():
    return psycopg.connect(_conninfo())


def load_user_profile(user_id: str) -> dict[str, Any] | None:
    """读取 L1 画像. 没有行返回 None."""
    if not _is_uuid(user_id):
        return None
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT interaction_style, spoiler_tolerance, humor_level,
                              confidence, messages_since_update
                       FROM user_profiles WHERE user_id = %s""",
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "interaction_style": row[0],
                    "spoiler_tolerance": row[1],
                    "humor_level": row[2],
                    "confidence": float(row[3] or 0),
                    "messages_since_update": int(row[4] or 0),
                }
    except Exception as e:
        print(f"[profile_store] load 失败: {e}", flush=True)
        return None


def render_profile_overlay(profile: dict[str, Any] | None) -> str:
    """把 L1 画像渲染成一行 system overlay. confidence 不够或没数据 → 空串.

    只渲染 style 类字段 (人设调整), 不渲染事实 — 避免 system prompt 变肥.
    """
    if not profile:
        return ""
    if float(profile.get("confidence", 0)) < PROFILE_INJECT_MIN_CONFIDENCE:
        return ""
    parts = []
    if profile.get("interaction_style"):
        parts.append(f"聊法偏好: {profile['interaction_style']}")
    if profile.get("spoiler_tolerance"):
        parts.append(f"剧透接受度: {profile['spoiler_tolerance']}")
    if profile.get("humor_level"):
        parts.append(f"接梗浓度: {profile['humor_level']}")
    if not parts:
        return ""
    return "（这位用户的长期偏好，请自然适应，不要复述这条说明: " + "，".join(parts) + "）"


def increment_message_counter(user_id: str) -> int:
    """累计一条非 refuse 对话, 返回累计后的计数 (用于判断是否触发更新).

    没有行则建一条默认行 (confidence=0)."""
    if not _is_uuid(user_id):
        return 0
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_profiles (user_id, messages_since_update)
                       VALUES (%s, 1)
                       ON CONFLICT (user_id) DO UPDATE
                         SET messages_since_update = user_profiles.messages_since_update + 1,
                             updated_at = now()
                       RETURNING messages_since_update""",
                    (user_id,),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0]) if row else 0
    except Exception as e:
        print(f"[profile_store] increment 失败: {e}", flush=True)
        return 0


def reset_message_counter(user_id: str) -> None:
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE user_profiles
                       SET messages_since_update = 0, updated_at = now()
                       WHERE user_id = %s""",
                    (user_id,),
                )
                conn.commit()
    except Exception as e:
        print(f"[profile_store] reset 失败: {e}", flush=True)


def save_profile(
    user_id: str,
    *,
    interaction_style: str | None,
    spoiler_tolerance: str | None,
    humor_level: str | None,
    confidence: float,
) -> None:
    """保存 L1 画像 (UPSERT)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_profiles
                         (user_id, interaction_style, spoiler_tolerance, humor_level,
                          confidence, messages_since_update, updated_at)
                       VALUES (%s, %s, %s, %s, %s, 0, now())
                       ON CONFLICT (user_id) DO UPDATE
                         SET interaction_style = EXCLUDED.interaction_style,
                             spoiler_tolerance = EXCLUDED.spoiler_tolerance,
                             humor_level       = EXCLUDED.humor_level,
                             confidence        = EXCLUDED.confidence,
                             messages_since_update = 0,
                             updated_at        = now()""",
                    (user_id, interaction_style, spoiler_tolerance, humor_level, confidence),
                )
                conn.commit()
    except Exception as e:
        print(f"[profile_store] save 失败: {e}", flush=True)


# ============================================================
# L2 作品画像 (per user × show)
# ============================================================

def load_show_profile(user_id: str, show: str) -> dict[str, Any] | None:
    """读取 L2 作品画像. 没有行返回 None."""
    if not _is_uuid(user_id) or not show:
        return None
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT favorite_characters, character_opinions, confidence
                       FROM show_profiles WHERE user_id = %s AND show = %s""",
                    (user_id, show),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "favorite_characters": list(row[0] or []),
                    "character_opinions": row[1] or [],
                    "confidence": float(row[2] or 0),
                }
    except Exception as e:
        print(f"[profile_store] load_show 失败: {e}", flush=True)
        return None


def save_show_profile(
    user_id: str,
    show: str,
    *,
    favorite_characters: list[str],
    character_opinions: list[dict],
    confidence: float,
) -> None:
    """保存 L2 作品画像 (UPSERT)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO show_profiles
                         (user_id, show, favorite_characters, character_opinions,
                          confidence, updated_at)
                       VALUES (%s, %s, %s, %s, %s, now())
                       ON CONFLICT (user_id, show) DO UPDATE
                         SET favorite_characters = EXCLUDED.favorite_characters,
                             character_opinions  = EXCLUDED.character_opinions,
                             confidence          = EXCLUDED.confidence,
                             updated_at           = now()""",
                    (
                        user_id, show,
                        list(favorite_characters or []),
                        Json(character_opinions or []),
                        confidence,
                    ),
                )
                conn.commit()
    except Exception as e:
        print(f"[profile_store] save_show 失败: {e}", flush=True)
