"""VideoLens 流式 ASR 服务 (常驻 WebSocket).

前端实时上传 PCM chunks, 转发给 DashScope paraformer-realtime-v2,
识别结果 partial/final 通过 ws 推回前端.

启动: python -m src.agent.asr_server
监听: ws://0.0.0.0:8000/stream
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import dashscope  # 真实 import 在 _init_session 里
import websockets
from dashscope.audio.asr import (
    Recognition,
    RecognitionCallback,
    RecognitionResult,
)

from src.core.config import get_config

logger = logging.getLogger("asr_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# DashScope 是同步 callback, 要把结果推回 asyncio 主 loop
# 用 run_coroutine_threadsafe 把 sync callback 的结果丢到 async queue


class ASRSession:
    """一次 WebSocket 连接 = 一次 ASR 会话.

    一个连接对应一个独立的 DashScope Recognition 实例.
    """

    def __init__(self, ws: websockets.WebSocketServerProtocol, loop: asyncio.AbstractEventLoop):
        self.ws = ws
        self.loop = loop
        self.recognition: Recognition | None = None
        self.result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = False
        self._feed_count = 0
        self._feed_bytes = 0
        self._stopped_once = False  # 防止 stop() 被调多次

    def _emit(self, payload: dict[str, Any]) -> None:
        """从 sync callback 跨线程把结果推到 async loop."""
        if self._closed:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.result_queue.put(payload),
                self.loop,
            )
        except Exception as e:
            logger.warning(f"_emit failed: {e}")

    def _on_event(self, result: RecognitionResult) -> None:
        """DashScope 识别事件 (sync thread). 区分 partial / final."""
        try:
            sentence = result.get_sentence()
            if not sentence or not isinstance(sentence, dict):
                logger.debug(f"_on_event: empty sentence ({type(sentence)})")
                return
            text = sentence.get("text", "")
            is_end = RecognitionResult.is_sentence_end(sentence)
            # 诊断: 任何事件都打日志 (看 DashScope 是否在返回数据)
            logger.info(
                f"_on_event: is_end={is_end} text={text!r} keys={list(sentence.keys())}"
            )
            if not text:
                return
            payload = {
                "type": "final" if is_end else "partial",
                "text": text,
            }
            self._emit(payload)
        except Exception as e:
            logger.warning(f"_on_event parse failed: {e}")

    def _on_error(self, result: Any) -> None:
        self._emit({"type": "error", "message": str(result)})

    def _on_close(self) -> None:
        self._emit({"type": "close"})

    def start(self) -> None:
        cfg = get_config()
        dashscope.api_key = cfg.dashscope_api_key

        session = self  # 闭包绑定

        class CB(RecognitionCallback):
            def on_event(self, result: RecognitionResult) -> None:
                session._on_event(result)

            def on_complete(self) -> None:
                session._emit({"type": "complete"})

            def on_error(self, result: Any) -> None:
                session._on_error(result)

            def on_close(self) -> None:
                session._on_close()

            def on_open(self) -> None:
                session._emit({"type": "ready"})

        self.recognition = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=16000,
            callback=CB(),
        )
        self.recognition.start()

    def feed(self, pcm_bytes: bytes) -> None:
        if not self.recognition:
            return
        try:
            # 诊断: 第一次 + 每 50 次打日志 (看前端是否真的在发音频)
            self._feed_count += 1
            self._feed_bytes += len(pcm_bytes)
            if self._feed_count == 1:
                logger.info(
                    f"first PCM chunk received: {len(pcm_bytes)} bytes"
                )
            elif self._feed_count % 50 == 0:
                logger.info(
                    f"received {self._feed_count} chunks, "
                    f"{self._feed_bytes} bytes total"
                )
            if hasattr(self.recognition, "stream_send_audio"):
                self.recognition.stream_send_audio(pcm_bytes)
            elif hasattr(self.recognition, "send_audio_frame"):
                self.recognition.send_audio_frame(pcm_bytes)
        except Exception as e:
            logger.warning(f"feed failed: {e}")

    def stop(self) -> None:
        # 防止重复 stop (handle_connection 最后总会调一次, 但 receive_audio
        # 收到 {action:stop} 时已经调过, 重复调报 "Speech recognition has stopped")
        if self._stopped_once:
            return
        self._stopped_once = True
        if self.recognition:
            try:
                self.recognition.stop()
            except Exception as e:
                logger.warning(f"stop failed: {e}")
        self._closed = True
        logger.info(
            f"session stopped: {self._feed_count} chunks, "
            f"{self._feed_bytes} bytes total fed"
        )


async def handle_connection(ws: websockets.WebSocketServerProtocol) -> None:
    """一次 WebSocket 连接的生命周期."""
    loop = asyncio.get_event_loop()
    session = ASRSession(ws, loop)
    logger.info(f"New ASR connection from {ws.remote_address}")

    try:
        session.start()
    except Exception as e:
        logger.error(f"DashScope session start failed: {e}")
        await ws.send(json.dumps({"type": "error", "message": f"start failed: {e}"}, ensure_ascii=False))
        return

    # 并发跑两个任务: 收音频 + 推识别结果
    async def receive_audio() -> None:
        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    session.feed(bytes(message))
                elif isinstance(message, str):
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("action") == "stop":
                        session.stop()
                        break
        except websockets.ConnectionClosed:
            pass

    async def send_results() -> None:
        try:
            while True:
                payload = await session.result_queue.get()
                await ws.send(json.dumps(payload, ensure_ascii=False))
                if payload.get("type") in {"close", "complete"}:
                    break
        except websockets.ConnectionClosed:
            pass

    await asyncio.gather(receive_audio(), send_results(), return_exceptions=True)
    session.stop()
    logger.info(f"Connection closed: {ws.remote_address}")


async def main() -> None:
    # 监听 0.0.0.0:8000 (本地 Next.js 通过 frp tunnel 反代过来)
    logger.info("ASR WebSocket server starting on ws://0.0.0.0:8000/stream")
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        8000,
        max_size=None,  # 允许大消息 (但实际 PCM chunk 一般 1-4KB)
        compression=None,  # 实时音频不需要压缩, 减少 CPU
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
