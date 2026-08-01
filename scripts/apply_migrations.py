"""按文件名顺序执行 db/migrations/*.sql, 让 schema 跟代码同步.

用法:
    python -m scripts.apply_migrations            # 用 AppConfig 的 postgres_url
    python -m scripts.apply_migrations --dry-run  # 只打印不执行

设计:
    - 所有 migration 文件必须用 IF NOT EXISTS / IF EXISTS 等幂等保护, 重复跑无副作用.
    - 不维护 schema_migrations 历史表 — 简化幂等性靠 SQL 本身.
    - 单个 migration 文件失败 → 立即终止后续 (避免顺序错乱).
    - init.sql 由 docker-compose 首次启动时自动跑 (这里不再重复跑).

适用场景:
    - 协作者 git pull 后, 跑一次同步本地已有库的 schema
    - 不适用于全新库 (全新库走 init.sql 自动初始化)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import psycopg

from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger()

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"

# 文件名规范: 数字前缀 (3-4 位) + 下划线 + 描述, 例: 0002_users_xxx.sql / 002_xxx.sql
_FILENAME_RE = re.compile(r"^\d+_.+\.sql$", re.IGNORECASE)


def _list_migrations() -> list[Path]:
    """按文件名排序返回所有 migration 文件 (符合命名规范的)."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    files = [p for p in MIGRATIONS_DIR.iterdir() if p.is_file() and _FILENAME_RE.match(p.name)]
    return sorted(files, key=lambda p: p.name)


def _run_one(conn: psycopg.Connection, path: Path, dry_run: bool) -> bool:
    """执行单个 migration 文件. 返回 True=成功/已应用, False=失败."""
    sql = path.read_text(encoding="utf-8")
    label = path.name
    if dry_run:
        print(f"  [DRY-RUN] {label} ({len(sql)} chars)")
        return True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"  [OK] {label}")
        return True
    except Exception as e:
        conn.rollback()
        print(f"  [FAIL] {label} -- {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="按顺序应用 db/migrations/*.sql")
    parser.add_argument("--dry-run", action="store_true", help="只列出文件不执行")
    args = parser.parse_args()

    cfg = get_config()
    if not cfg.postgres_url:
        print("[ERROR] 未配置 postgres_url (POSTGRES_URL 环境变量)", file=sys.stderr)
        return 1

    migrations = _list_migrations()
    if not migrations:
        print(f"[INFO] migrations 目录为空或不存在的路径, 跳过")

    print(f"目标库: {_mask_url(cfg.postgres_url)}")
    print(f"待执行 ({len(migrations)} 个文件):")
    for p in migrations:
        print(f"  - {p.name}")
    print()

    if args.dry_run:
        print("[DRY-RUN] 不执行")
        for p in migrations:
            _run_one(None, p, dry_run=True)
        return 0

    try:
        with psycopg.connect(cfg.postgres_url) as conn:
            for p in migrations:
                ok = _run_one(conn, p, dry_run=False)
                if not ok:
                    print(f"\n[FAIL] 在 {p.name} 处终止, 后续未执行", file=sys.stderr)
                    return 1
    except psycopg.OperationalError as e:
        print(f"[ERROR] 连不上 Postgres: {e}", file=sys.stderr)
        print("  检查 POSTGRES_URL 或确认 docker-compose up -d 已起", file=sys.stderr)
        return 1

    print(f"\n[OK] 全部完成 ({len(migrations)} 个文件)")
    return 0


def _mask_url(url: str) -> str:
    """隐藏密码, 只显示 host/db."""
    m = re.match(r"postgresql?://([^:]+):([^@]+)@([^/]+)/(.+)", url)
    if not m:
        return url
    user, _, host, db = m.groups()
    return f"postgresql://{user}:***@{host}/{db}"


if __name__ == "__main__":
    sys.exit(main())
