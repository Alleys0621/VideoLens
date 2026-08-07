"""人设卡配置与用户/thread 选择读取."""

from __future__ import annotations

from typing import Any

import psycopg

from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger()

DEFAULT_PERSONA_ID = "alleys"


def get_persona_meta(persona_id: str | None) -> dict[str, Any]:
    persona_id = persona_id or DEFAULT_PERSONA_ID
    cfg = get_config()
    personas = (cfg.prompts.get("companion_prompts", {}) or {}).get("personas", {}) or {}
    meta = personas.get(persona_id)
    if not isinstance(meta, dict):
        persona_id = DEFAULT_PERSONA_ID
        meta = personas.get(DEFAULT_PERSONA_ID, {}) or {}
    return {
        "id": persona_id,
        "name": meta.get("name", "小艾"),
        "tagline": meta.get("tagline", "温柔知心"),
        "system": meta.get("system", "") or "",
    }


def build_persona_system_prompt(persona_id: str | None) -> str:
    cfg = get_config()
    prompts = cfg.prompts.get("companion_prompts", {}) or {}
    base = prompts.get("system_base", "") or ""
    persona = get_persona_meta(persona_id)
    parts = [p for p in (persona["system"], base) if p.strip()]
    return "\n\n".join(parts)


def _is_uuid(value: str | None) -> bool:
    if not value:
        return False
    import re
    return bool(re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    ))


def load_default_persona_id(user_id: str | None) -> str:
    if not _is_uuid(user_id):
        return DEFAULT_PERSONA_ID
    try:
        with psycopg.connect(get_config().postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT default_persona_id FROM user_preferences WHERE user_id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return row[0]
    except Exception as e:
        logger.warning(f"[persona_store] load default persona 失败: {e}")
    return DEFAULT_PERSONA_ID


def load_thread_persona_id(thread_id: str | None, user_id: str | None) -> str:
    if not thread_id:
        return load_default_persona_id(user_id)
    try:
        with psycopg.connect(get_config().postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT persona_id FROM threads WHERE thread_id = %s",
                    (thread_id,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return row[0]
    except Exception as e:
        logger.warning(f"[persona_store] load thread persona 失败: {e}")
    return load_default_persona_id(user_id)
