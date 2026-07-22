"""VideoLens 流式 TTS 服务 (常驻 WebSocket).

一个前端 ws 会话 = 一个 DashScope task. 前端发 start/text/finish,
本服务桥接到 DashScope, mp3 二进制原样转发回前端.

模型: qwen-audio-3.0-tts-flash + longanhuan_v3.6 (普通话女声)
启动: python -m src.agent.tts_server
监听: ws://0.0.0.0:8001/
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import websockets
from websockets.asyncio.client import connect as ws_connect

from src.core.config import get_config

logger = logging.getLogger("tts_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_DASHSCOPE_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"

_MODEL = "qwen-audio-3.0-tts-flash"
_VOICE = "longanhuan_v3.6"
_FORMAT = "mp3"
_SAMPLE_RATE = 24000

# DashScope 连接 / 任务超时
_DS_OPEN_TIMEOUT = 10
_TASK_IDLE_TIMEOUT = 60  # task 启动后 60s 内没有任何活动就关


async def _frontend_to_dashscope(
    frontend_ws,
    ds_ws,
    task_id: str,
    started_event: asyncio.Event,
) -> None:
    """前端 ws → DashScope ws: 转发 text / finish.

    流程:
      1. 等前端发 start → 我们已在外层处理 (建立 ds_ws + run-task)
      2. 收前端 text → 转 continue-task
      3. 收前端 finish → 转 finish-task (然后退出)
      4. 收前端 stop → 直接 close ds_ws
    """
    try:
        async for raw in frontend_ws:
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")

            if mtype == "text":
                text = msg.get("text") or ""
                if not text:
                    continue
                # 等 task-started 才能发 continue-task
                await started_event.wait()
                await ds_ws.send(
                    json.dumps(
                        {
                            "header": {
                                "action": "continue-task",
                                "task_id": task_id,
                                "streaming": "duplex",
                            },
                            "payload": {"input": {"text": text}},
                        },
                        ensure_ascii=False,
                    )
                )
            elif mtype == "finish":
                # 等 task-started 才能发 finish-task
                await started_event.wait()
                await ds_ws.send(
                    json.dumps(
                        {
                            "header": {
                                "action": "finish-task",
                                "task_id": task_id,
                                "streaming": "duplex",
                            },
                            "payload": {"input": {}},
                        },
                        ensure_ascii=False,
                    )
                )
                # finish 发完, 这个方向退出 (等 ds_to_frontend 处理 task-finished)
                return
            elif mtype == "stop":
                # 用户中断, 直接关 ds_ws
                logger.info(f"[{task_id}] frontend requested stop")
                await ds_ws.close()
                return
            elif mtype == "ping":
                await frontend_ws.send(json.dumps({"type": "pong"}))
    except websockets.ConnectionClosed:
        pass


async def _dashscope_to_frontend(
    ds_ws,
    frontend_ws,
    task_id: str,
    started_event: asyncio.Event,
) -> None:
    """DashScope ws → 前端 ws: 推 mp3 二进制 + 转换 JSON 事件.

    流程:
      1. 收 task-started → set started_event + 发 {type: "ready"}
      2. 收 result-generated (二进制 mp3 chunk) → 原样转发
      3. 收 task-finished → 发 {type: "complete"} + 退出
      4. 收 task-failed → 发 {type: "error", message}
    """
    chunk_count = 0
    byte_count = 0
    try:
        async for raw in ds_ws:
            if isinstance(raw, (bytes, bytearray)):
                chunk_count += 1
                byte_count += len(raw)
                await frontend_ws.send(raw)
            elif isinstance(raw, str):
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event = msg.get("header", {}).get("event", "")
                if event == "task-started":
                    started_event.set()
                    await frontend_ws.send(json.dumps({"type": "ready"}))
                elif event == "task-finished":
                    logger.info(
                        f"[{task_id}] task-finished: "
                        f"{chunk_count} chunks, {byte_count} bytes"
                    )
                    await frontend_ws.send(json.dumps({"type": "complete"}))
                    return
                elif event == "task-failed":
                    err = (
                        msg.get("header", {}).get("error_message")
                        or "task-failed"
                    )
                    logger.error(f"[{task_id}] task-failed: {err}")
                    await frontend_ws.send(
                        json.dumps(
                            {"type": "error", "message": f"DashScope: {err}"},
                            ensure_ascii=False,
                        )
                    )
                    return
    except websockets.ConnectionClosed:
        pass


async def _serve_session(frontend_ws) -> None:
    """一次前端 ws 连接 = 一个 DashScope task 生命周期.

    协议:
      前端必须先发 {type: "start"} 才能发 text/finish/stop.
      本函数收到 start 后建立 DashScope ws, 启动双向桥接.
    """
    peer = getattr(frontend_ws, "remote_address", None)
    logger.info(f"Frontend connected: {peer}")

    # 等前端发 start
    try:
        async for raw in frontend_ws:
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "start":
                break
            elif msg.get("type") == "ping":
                await frontend_ws.send(json.dumps({"type": "pong"}))
            # 其他消息在 start 之前忽略
    except websockets.ConnectionClosed:
        logger.info(f"Frontend disconnected before start: {peer}")
        return

    # 建立 DashScope ws + 发 run-task
    cfg = get_config()
    api_key = cfg.dashscope_api_key
    if not api_key:
        await frontend_ws.send(
            json.dumps(
                {"type": "error", "message": "DASHSCOPE_API_KEY 未配置"},
                ensure_ascii=False,
            )
        )
        return

    headers = {
        "Authorization": f"bearer {api_key}",
        "X-DashScope-DataInspection": "enable",
    }
    task_id = str(uuid.uuid4())

    try:
        async with ws_connect(
            _DASHSCOPE_WS_URL,
            additional_headers=headers,
            max_size=None,
            open_timeout=_DS_OPEN_TIMEOUT,
            close_timeout=5,
        ) as ds_ws:
            # 发 run-task
            await ds_ws.send(
                json.dumps(
                    {
                        "header": {
                            "action": "run-task",
                            "task_id": task_id,
                            "streaming": "duplex",
                        },
                        "payload": {
                            "task_group": "audio",
                            "task": "tts",
                            "function": "SpeechSynthesizer",
                            "model": _MODEL,
                            "parameters": {
                                "text_type": "PlainText",
                                "voice": _VOICE,
                                "format": _FORMAT,
                                "sample_rate": _SAMPLE_RATE,
                                "volume": 50,
                                "rate": 1,
                                "pitch": 1,
                                # False 允许发多次 continue-task (LLM 流式 token)
                                "enable_ssml": False,
                            },
                            "input": {},
                        },
                    },
                    ensure_ascii=False,
                )
            )

            started_event = asyncio.Event()

            # 双向桥接 (并发)
            ft = asyncio.create_task(
                _frontend_to_dashscope(frontend_ws, ds_ws, task_id, started_event),
                name=f"fe→ds-{task_id[:8]}",
            )
            dt = asyncio.create_task(
                _dashscope_to_frontend(ds_ws, frontend_ws, task_id, started_event),
                name=f"ds→fe-{task_id[:8]}",
            )

            done, pending = await asyncio.wait(
                {ft, dt},
                return_when=asyncio.ALL_COMPLETED,
                timeout=_TASK_IDLE_TIMEOUT,
            )
            for t in pending:
                t.cancel()
    except asyncio.TimeoutError:
        logger.error(f"[{task_id}] dashscope connect timeout")
        try:
            await frontend_ws.send(
                json.dumps(
                    {"type": "error", "message": "DashScope 连接超时"},
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass
    except Exception as e:
        logger.exception(f"[{task_id}] session error")
        try:
            await frontend_ws.send(
                json.dumps(
                    {"type": "error", "message": f"内部错误: {e}"},
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass
    finally:
        logger.info(f"Frontend disconnected: {peer}")


async def main() -> None:
    logger.info("TTS WebSocket server starting on ws://0.0.0.0:9801/")
    logger.info(f"  model={_MODEL}  voice={_VOICE}  format={_FORMAT}  sr={_SAMPLE_RATE}")
    async with websockets.serve(
        _serve_session,
        "0.0.0.0",
        9801,
        max_size=None,
        compression=None,
        ping_interval=20,
        ping_timeout=10,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
