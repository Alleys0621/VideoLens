"""清理 LangGraph dev store 里的孤儿 thread (无 user_id metadata 或 state.messages 为空).

用法:
    python -m scripts.cleanup_orphan_threads

行为:
    1. 列出所有 thread, 标记 orphan (metadata.user_id 缺失/default 或 state.messages 空)
    2. 对每个 orphan 调 DELETE /threads/{tid} 删除 LangGraph 那边的 thread + state
    3. 同时从 PG threads 表删 (如果有)
"""

from __future__ import annotations

import json
import os
import urllib.request

LANGGRAPH_URL = "http://127.0.0.1:2024"
PG_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens",
)


def _list_threads() -> list[dict]:
    req = urllib.request.Request(
        f"{LANGGRAPH_URL}/threads/search",
        data=json.dumps({"limit": 200}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=10))


def _get_state(tid: str) -> dict:
    req = urllib.request.Request(f"{LANGGRAPH_URL}/threads/{tid}/state")
    return json.load(urllib.request.urlopen(req, timeout=5))


def _delete_thread(tid: str) -> bool:
    req = urllib.request.Request(
        f"{LANGGRAPH_URL}/threads/{tid}", method="DELETE"
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  ! delete {tid[:8]} failed: {e}")
        return False


def main():
    print("=" * 60)
    print("VideoLens 孤儿 thread 清理")
    print("=" * 60)
    threads = _list_threads()
    print(f"扫描 {len(threads)} 个 thread...")

    orphans: list[str] = []
    for t in threads:
        tid = t["thread_id"]
        meta = t.get("metadata") or {}
        uid = meta.get("user_id")
        if not uid or uid == "default" or uid == "MISSING":
            orphans.append(tid)
            print(f"  orphan (no user_id): {tid[:8]}")
            continue
        try:
            st = _get_state(tid)
            msgs = (st.get("values") or {}).get("messages") or []
            if len(msgs) == 0:
                orphans.append(tid)
                print(f"  orphan (empty state): {tid[:8]}")
        except Exception as e:
            print(f"  ! state fetch {tid[:8]} failed: {e}")
            orphans.append(tid)

    print(f"\n发现 {len(orphans)} 个孤儿 thread")

    if not orphans:
        print("没有需要清理的 thread")
        return

    confirm = input("\n确认清理? (y/N): ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    n_ok = 0
    for tid in orphans:
        if _delete_thread(tid):
            n_ok += 1
            print(f"  ✓ deleted {tid[:8]}")

    # 从 PG 也删 (如果有残留)
    try:
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "DELETE FROM threads WHERE thread_id = %s",
                    [(tid,) for tid in orphans],
                )
            conn.commit()
        print(f"\n  PG threads 表清理完成")
    except Exception as e:
        print(f"  PG cleanup skipped: {e}")

    print(f"\n完成: 删除 {n_ok}/{len(orphans)} 个孤儿 thread")


if __name__ == "__main__":
    main()
