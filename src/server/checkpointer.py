"""LangGraph Postgres checkpointer factory.

`langgraph.json` 通过 `checkpointer.path` 引用本模块的 `create_checkpointer`
async context manager, 使 dev server 启动时挂载 Postgres 持久化 checkpointer,
thread state (messages/checkpoints) 写入 Postgres, 跨重启保留.

连接串从环境变量 POSTGRES_URL / DATABASE_URI 读取.
首次连接会自动建 checkpoints / checkpoint_writes / checkpoint_blobs 表 (saver.setup()).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def _get_conninfo() -> str:
    """读取 Postgres 连接串.

    优先级: POSTGRES_URL > DATABASE_URI > 默认本地 dev 连接.
    默认值与 db/docker-compose.yml 一致 (videolens / videolens_dev / videolens).
    """
    return (
        os.getenv("POSTGRES_URL")
        or os.getenv("DATABASE_URI")
        or "postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens"
    )


@asynccontextmanager
async def create_checkpointer():
    """创建 Postgres checkpointer 单例.

    AsyncPostgresSaver.from_conn_string 内部用 psycopg pool 管理连接,
    自动调用 setup() 建表, yield 后退出时关闭 pool.
    """
    conninfo = _get_conninfo()
    async with AsyncPostgresSaver.from_conn_string(conninfo) as saver:
        # 首次连接时建表 (已存在则跳过, 幂等)
        await saver.setup()
        yield saver
