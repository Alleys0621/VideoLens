"""一次性迁移脚本: 从 .langgraph_api/store.pckl 把老用户/播放进度导入 Postgres.

用法:
    # 启动 docker compose (Postgres) 后, 项目根目录运行
    python -m scripts.migrate_to_postgres

行为:
    - 读取 .langgraph_api/store.pckl (LangGraph dev 的 InMemoryStore pickle)
    - 提取 namespace=["users"] 的用户记录 → INSERT 进 users 表 (ON CONFLICT 跳过)
    - 提取 namespace=["playback", user_id] 的播放进度 → INSERT 进 playback_progress 表
    - 同时调用 LangGraph API 列出所有 thread, 把 metadata.user_id 完整的 thread 同步到 PG
      state.messages 为空的 thread 视为"孤儿", 列出来供人工确认删除

幂等: ON CONFLICT DO NOTHING, 重跑不会重复插入.
"""

from __future__ import annotations

import os
import pickle
import sys
import uuid
from pathlib import Path

import psycopg


# ---- 配置 ----
PG_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens",
)
STORE_PCKL = Path(".langgraph_api/store.pckl")
LANGGRAPH_URL = "http://127.0.0.1:2024"


def _load_store() -> dict:
    """加载 LangGraph dev store pickle."""
    if not STORE_PCKL.is_file():
        print(f"[migrate] {STORE_PCKL} not found, nothing to migrate")
        return {}
    with STORE_PCKL.open("rb") as f:
        return pickle.load(f)


def _migrate_users(store: dict, conn) -> int:
    """迁移 namespace=["users"] 的记录 → users 表."""
    # store 结构: dict[(namespace_tuple, key)] = value_dict
    # 或: {namespace_tuple: {key: value}}
    count = 0
    for key, value in _iter_namespace(store, ("users",)):
        if not isinstance(value, dict):
            continue
        phone = value.get("phone")
        pwd_hash = value.get("password_hash")
        display = value.get("display_name") or f"用户_{str(phone)[-4:]}"
        created = value.get("created_at") or "2020-01-01T00:00:00Z"
        if not phone or not pwd_hash:
            continue
        # key 通常是 UUID 字符串; 校验一下
        try:
            uid = str(uuid.UUID(str(key)))
        except (ValueError, TypeError):
            uid = str(uuid.uuid4())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, phone, password_hash, display_name, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (uid, phone, pwd_hash, display, created),
            )
            if cur.rowcount > 0:
                count += 1
                print(f"  + user {uid[:8]} phone={phone} name={display}")
    return count


def _migrate_playback(store: dict, conn) -> int:
    """迁移 namespace=["playback", user_id] → playback_progress 表."""
    count = 0
    for (ns, key), value in _iter_namespace_kv(store, prefix=("playback",)):
        if not isinstance(value, dict):
            continue
        # ns = ("playback", user_id)
        if len(ns) < 2:
            continue
        user_id = ns[1]
        try:
            user_uuid = str(uuid.UUID(str(user_id)))
        except (ValueError, TypeError):
            print(f"  ! skip playback for non-uuid user_id={user_id}")
            continue
        video_dir = key
        position = float(value.get("position") or 0)
        duration = value.get("duration")
        completed = bool(value.get("completed"))
        updated = value.get("updated_at") or "2020-01-01T00:00:00Z"
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO playback_progress
                  (user_id, video_dir, position, duration, completed, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, video_dir) DO NOTHING
                """,
                (user_uuid, video_dir, position, duration, completed, updated),
            )
            if cur.rowcount > 0:
                count += 1
    print(f"  migrated {count} playback_progress rows")
    return count


def _migrate_threads(conn) -> tuple[int, list[str]]:
    """从 LangGraph API 列所有 thread, 同步 metadata.user_id 完整的到 PG.
    返回 (migrated_count, orphan_thread_ids) — orphan 是 metadata 缺 user_id 或 state 空的.
    """
    import urllib.request
    import json

    print("\n[migrate] scanning LangGraph threads...")
    req = urllib.request.Request(
        f"{LANGGRAPH_URL}/threads/search",
        data=json.dumps({"limit": 200}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        threads = json.load(urllib.request.urlopen(req, timeout=10))
    except Exception as e:
        print(f"  ! LangGraph API 调用失败: {e}")
        return 0, []

    migrated = 0
    orphans = []
    for t in threads:
        tid = t.get("thread_id")
        meta = t.get("metadata") or {}
        uid = meta.get("user_id")
        if not uid or uid == "default" or uid == "MISSING":
            orphans.append(tid)
            continue
        try:
            user_uuid = str(uuid.UUID(str(uid)))
        except (ValueError, TypeError):
            orphans.append(tid)
            continue
        # 检查 state 是否空
        try:
            req2 = urllib.request.Request(f"{LANGGRAPH_URL}/threads/{tid}/state")
            st = json.load(urllib.request.urlopen(req2, timeout=5))
            msgs = (st.get("values") or {}).get("messages") or []
            if len(msgs) == 0:
                orphans.append(tid)
                continue
        except Exception:
            orphans.append(tid)
            continue

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO threads (thread_id, user_id)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (tid, user_uuid),
            )
            if cur.rowcount > 0:
                migrated += 1
                print(f"  + thread {tid[:8]} user={user_uuid[:8]}")
    print(f"  migrated {migrated} threads; {len(orphans)} orphans identified")
    return migrated, orphans


def _unwrap_item(raw):
    """LangGraph InMemoryStore 的 value 可能是 Item 对象, 解出真正的 value dict."""
    if isinstance(raw, dict):
        return raw
    # Item(namespace=..., key=..., value={...}, created_at=..., updated_at=...)
    if hasattr(raw, "value") and isinstance(getattr(raw, "value"), dict):
        return getattr(raw, "value")
    return None


def _item_key(raw, fallback):
    """从 Item 对象取 key, 否则用外层 dict key."""
    if hasattr(raw, "key"):
        return getattr(raw, "key") or fallback
    return fallback


def _iter_namespace(store: dict, ns: tuple):
    """兼容两种 store pickle 结构, yield (key, value_dict) for given namespace."""
    # 结构 A: {("ns",): {key: Item/value}}
    if ns in store and isinstance(store[ns], dict):
        for k, v in store[ns].items():
            val = _unwrap_item(v)
            if val is not None:
                yield _item_key(v, k), val
        return
    # 结构 B: 顶层 dict, key 包含 namespace
    # 留作扩展 (实测当前是结构 A)


def _iter_namespace_kv(store: dict, prefix: tuple):
    """yield ((namespace_tuple, key), value_dict) for all namespaces starting with prefix."""
    for ns_key, value in store.items():
        if not isinstance(ns_key, tuple) or len(ns_key) == 0:
            continue
        if ns_key[: len(prefix)] != prefix:
            continue
        if isinstance(value, dict):
            for k, v in value.items():
                val = _unwrap_item(v)
                if val is not None:
                    yield (ns_key, _item_key(v, k)), val
        else:
            val = _unwrap_item(value)
            if val is not None:
                yield (ns_key, _item_key(value, "")), val


def main():
    print("=" * 60)
    print("VideoLens 数据迁移: store.pckl → Postgres")
    print(f"  PG_URL  = {PG_URL}")
    print(f"  STORE   = {STORE_PCKL}")
    print("=" * 60)

    if not STORE_PCKL.is_file():
        print(f"[skip] {STORE_PCKL} 不存在, 无需迁移")
        sys.exit(0)

    store = _load_store()
    print(f"[load] store keys: {len(store) if hasattr(store, '__len__') else '?'}")

    with psycopg.connect(PG_URL) as conn:
        with conn.transaction():  # transaction
            print("\n[migrate] users...")
            n_users = _migrate_users(store, conn)
            print(f"  → {n_users} users inserted")

            print("\n[migrate] threads (from LangGraph API)...")
            n_threads, orphans = _migrate_threads(conn)

        # playback 用单独 transaction
        with conn.transaction():
            print("\n[migrate] playback_progress...")
            n_play = _migrate_playback(store, conn)

    print("\n" + "=" * 60)
    print(f"迁移完成: {n_users} users / {n_threads} threads / {n_play} playback")
    if orphans:
        print(f"\n[清理建议] {len(orphans)} 个孤儿 thread (无 user_id 或空 state):")
        for tid in orphans[:20]:
            print(f"  - {tid}")
        if len(orphans) > 20:
            print(f"  ... 还有 {len(orphans) - 20} 个")
        print("\n建议运行清理脚本: python -m scripts.cleanup_orphan_threads")
    print("=" * 60)


if __name__ == "__main__":
    main()
