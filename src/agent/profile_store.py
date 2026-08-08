"""L1 用户长期画像读写 (Postgres).

字段按稳定性分两层:
  内核层 (稳定, 不易变): interaction_style / interaction_initiative /
                        engagement_motivation / humor_level / teasing_tolerance /
                        spoiler_tolerance / pet_peeves
    - 完整轨道每 PROFILE_UPDATE_THRESHOLD 轮更新, conf_stable 用 EWMA 维护,
      渲染时 conf_stable>=0.6 才注入.
  表现层 (易变, 用户当下指令): alleys_attitude / alleys_response_preference
    - 轻量轨道每轮更新 (update_instant_fields), 直接覆盖无置信度,
      渲染时始终注入并排第一位.

职责边界:
  - 本表: "这个人什么样" (结构化画像) → 注入 user prompt.
  - Mem0: 零散事实回忆 (喜欢刘星 / 上次聊到 XX) → context 检索.
  两者不重叠.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Json

from src.core.logging import get_logger

logger = get_logger()


PROFILE_UPDATE_THRESHOLD = 5  # 完整轨道每累计 5 条对话触发一次 (内核层, 慢更新)


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
                              engagement_motivation, interaction_initiative,
                              teasing_tolerance, pet_peeves, conf_stable,
                              alleys_attitude, alleys_response_preference,
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
                    "interaction_initiative": row[4],
                    "teasing_tolerance": row[5],
                    "pet_peeves": list(row[6] or []),
                    "conf_stable": float(row[7] or 0),
                    "alleys_attitude": row[8],
                    "alleys_response_preference": row[9],
                    "messages_since_update": int(row[10] or 0),
                }
    except Exception as e:
        logger.warning(f"[profile_store] load 失败: {e}")
        return None


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
    interaction_initiative: str | None = None,
    teasing_tolerance: str | None = None,
    pet_peeves: list[str] | None = None,
    alleys_attitude: str | None = None,
    alleys_response_preference: str | None = None,
    conf_stable: float,
) -> None:
    """保存 L1 画像 (完整轨道, UPSERT 全部字段)."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_profiles
                         (user_id, interaction_style, spoiler_tolerance, humor_level,
                          engagement_motivation, interaction_initiative,
                          teasing_tolerance, pet_peeves,
                          alleys_attitude, alleys_response_preference,
                          conf_stable, messages_since_update, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, now())
                       ON CONFLICT (user_id) DO UPDATE
                         SET interaction_style          = EXCLUDED.interaction_style,
                             spoiler_tolerance          = EXCLUDED.spoiler_tolerance,
                             humor_level                = EXCLUDED.humor_level,
                             engagement_motivation      = EXCLUDED.engagement_motivation,
                             interaction_initiative     = EXCLUDED.interaction_initiative,
                             teasing_tolerance          = EXCLUDED.teasing_tolerance,
                             pet_peeves                 = EXCLUDED.pet_peeves,
                             alleys_attitude            = EXCLUDED.alleys_attitude,
                             alleys_response_preference = EXCLUDED.alleys_response_preference,
                             conf_stable                = EXCLUDED.conf_stable,
                             updated_at                 = now()""",
                    (user_id, interaction_style, spoiler_tolerance, humor_level,
                     engagement_motivation, interaction_initiative,
                     teasing_tolerance, list(pet_peeves or []),
                     alleys_attitude, alleys_response_preference,
                     conf_stable),
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"[profile_store] save 失败: {e}")


def update_instant_fields(
    user_id: str,
    alleys_attitude: str | None,
    alleys_response_preference: str | None,
) -> None:
    """轻量轨道: 只写表现层两字段 (用户当下指令), 直接覆盖, 无置信度.

    UPSERT 兼容首次无画像行 (冷启动首轮即建行, 其他字段用默认值).
    """
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_profiles
                         (user_id, alleys_attitude, alleys_response_preference, updated_at)
                       VALUES (%s, %s, %s, now())
                       ON CONFLICT (user_id) DO UPDATE
                         SET alleys_attitude            = EXCLUDED.alleys_attitude,
                             alleys_response_preference = EXCLUDED.alleys_response_preference,
                             updated_at                 = now()""",
                    (user_id, alleys_attitude, alleys_response_preference),
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"[profile_store] update_instant 失败: {e}")


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
