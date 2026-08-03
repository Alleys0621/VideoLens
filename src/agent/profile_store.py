"""L1 用户长期画像读写 (Postgres).

存储结构化、可注入、可审计的人设级偏好 (聊法 / 剧透接受度 / 接梗浓度).
字段值用中文, DBeaver 直接可读.

职责边界:
  - 本表: "这个人什么样" (style 类) → 作为 system overlay 注入.
  - Mem0: 零散事实回忆 (喜欢刘星 / 上次聊到 XX) → 作为 context 检索.
  两者不重叠.

不每轮写: 由 companion 累加 messages_since_update,
达到 PROFILE_UPDATE_THRESHOLD 才增量更新 (见 profile_updater).
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Json

from src.core.logging import get_logger

logger = get_logger()


PROFILE_UPDATE_THRESHOLD = 2  # 每累计 2 条对话触发一次增量更新


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
                              engagement_motivation, alleys_attitude, confidence,
                              messages_since_update
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
                    "engagement_motivation": row[3],
                    "alleys_attitude": row[4],
                    "confidence": float(row[5] or 0),
                    "messages_since_update": int(row[6] or 0),
                }
    except Exception as e:
        logger.warning(f"[profile_store] load 失败: {e}")
        return None


def render_profile_overlay(profile: dict[str, Any] | None) -> str:
    """把 L1 画像渲染成一行 system overlay. confidence 不够或没数据 → 空串.

    只渲染 style 类字段 (人设调整), 不渲染事实 — 避免 system prompt 变肥.
    """
    if not profile:
        return ""
    parts = []
    if profile.get("interaction_style"):
        parts.append(f"聊法偏好: {profile['interaction_style']}")
    if profile.get("spoiler_tolerance"):
        parts.append(f"剧透接受度: {profile['spoiler_tolerance']}")
    if profile.get("humor_level"):
        parts.append(f"接梗浓度: {profile['humor_level']}")
    if profile.get("engagement_motivation"):
        parts.append(f"观看动力: {profile['engagement_motivation']}")
    attitude = profile.get("alleys_attitude")
    if attitude:
        parts.append(f"用户对你的态度: {attitude}")
    if not parts:
        return ""
    return (
        "（这位用户的偏好只用来调整你的语气，"
        "**不要因此多反问**（即使偏好写'提问型', 也是用户喜欢刨根问底, 不是你要多反问）"
        "，同时请认真对待'用户对你的态度'——它直接告诉你用户希望你以什么方式回应: "
        + "，".join(parts) + "）"
    )


def increment_message_counter(user_id: str) -> bool:
    """累加对话计数; 达阈值原子归零返回 True. 归零在同一 SQL, 防 async 并发重复触发."""
    if not _is_uuid(user_id):
        return False
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_profiles (user_id, messages_since_update)
                       VALUES (%s, 1)
                       ON CONFLICT (user_id) DO UPDATE
                         SET messages_since_update =
                               CASE WHEN user_profiles.messages_since_update + 1 >= %s
                                    THEN 0
                                    ELSE user_profiles.messages_since_update + 1 END
                       RETURNING messages_since_update = 0 AS triggered""",
                    (user_id, PROFILE_UPDATE_THRESHOLD),
                )
                row = cur.fetchone()
                conn.commit()
                return bool(row[0]) if row else False
    except Exception as e:
        logger.warning(f"[profile_store] increment 失败: {e}")
        return False


def save_profile(
    user_id: str,
    *,
    interaction_style: str | None,
    spoiler_tolerance: str | None,
    humor_level: str | None,
    engagement_motivation: str | None,
    alleys_attitude: str | None = None,
    confidence: float,
) -> None:
    """保存 L1 画像 (UPSERT)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_profiles
                         (user_id, interaction_style, spoiler_tolerance, humor_level,
                          engagement_motivation, alleys_attitude, confidence,
                          messages_since_update, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, 0, now())
                       ON CONFLICT (user_id) DO UPDATE
                         SET interaction_style     = EXCLUDED.interaction_style,
                             spoiler_tolerance     = EXCLUDED.spoiler_tolerance,
                             humor_level           = EXCLUDED.humor_level,
                             engagement_motivation = EXCLUDED.engagement_motivation,
                             alleys_attitude       = EXCLUDED.alleys_attitude,
                             confidence            = EXCLUDED.confidence,
                             updated_at            = now()""",
                    (user_id, interaction_style, spoiler_tolerance, humor_level,
                     engagement_motivation, alleys_attitude, confidence),
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"[profile_store] save 失败: {e}")


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
                    """SELECT favorite_characters, attention_characters,
                              character_opinions, theme_preferences, disliked_elements,
                              confidence
                       FROM show_profiles WHERE user_id = %s AND show = %s""",
                    (user_id, show),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "favorite_characters": list(row[0] or []),
                    "attention_characters": list(row[1] or []),
                    "character_opinions": row[2] or [],
                    "theme_preferences": list(row[3] or []),
                    "disliked_elements": list(row[4] or []),
                    "confidence": float(row[5] or 0),
                }
    except Exception as e:
        logger.warning(f"[profile_store] load_show 失败: {e}")
        return None


def save_show_profile(
    user_id: str,
    show: str,
    *,
    favorite_characters: list[str],
    attention_characters: list[str],
    character_opinions: list[dict],
    theme_preferences: list[str],
    disliked_elements: list[str],
    confidence: float,
) -> None:
    """保存 L2 作品画像 (UPSERT)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO show_profiles
                         (user_id, show, favorite_characters, attention_characters,
                          character_opinions, theme_preferences, disliked_elements,
                          confidence, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                       ON CONFLICT (user_id, show) DO UPDATE
                         SET favorite_characters  = EXCLUDED.favorite_characters,
                             attention_characters = EXCLUDED.attention_characters,
                             character_opinions   = EXCLUDED.character_opinions,
                             theme_preferences    = EXCLUDED.theme_preferences,
                             disliked_elements    = EXCLUDED.disliked_elements,
                             confidence           = EXCLUDED.confidence,
                             updated_at            = now()""",
                    (
                        user_id, show,
                        list(favorite_characters or []),
                        list(attention_characters or []),
                        Json(character_opinions or []),
                        list(theme_preferences or []),
                        list(disliked_elements or []),
                        confidence,
                    ),
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"[profile_store] save_show 失败: {e}")
