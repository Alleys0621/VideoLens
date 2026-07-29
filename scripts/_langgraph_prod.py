"""生产模式 LangGraph 启动入口 (Windows).

用途: 在 Windows 上让 `langgraph dev --no-reload` 跑通. 直接 langgraph dev --no-reload
会触发 ProactorEventLoop 错误 (uvicorn 主进程默认用 Proactor, psycopg async 需要 Selector).

本 wrapper 在 cli 调用前 monkey-patch uvicorn 的 loop factory, 强制 SelectorEventLoop.

仅供 start-prod.bat 调用, 不进日常 dev 流程 (日常 dev 用 langgraph dev, 享受热更新).

用法:
    python -m scripts._langgraph_prod dev --port 2024 --no-browser --no-reload
"""
from __future__ import annotations

import sys


def _patch_uvicorn_loop() -> None:
    """强制 uvicorn 用 SelectorEventLoop (Windows 上 psycopg async 需要).

    uvicorn/loops/asyncio.py 默认在 win32 主进程返回 ProactorEventLoop,
    会让 psycopg async 报 InterfaceError. patch 后始终返回 SelectorEventLoop.
    """
    if sys.platform != "win32":
        return
    import asyncio
    import uvicorn.loops.asyncio as _uvloop

    def _selector_loop_factory(use_subprocess: bool = False):  # noqa: ARG001
        return asyncio.SelectorEventLoop

    _uvloop.asyncio_loop_factory = _selector_loop_factory


def main() -> None:
    _patch_uvicorn_loop()
    from langgraph_cli.cli import cli
    cli()


if __name__ == "__main__":
    main()
