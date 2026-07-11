"""ASR 服务: paraformer-realtime-v2 (便宜, 不用 omni), fallback qwen-omni-flash.

stdin: wav bytes → stdout: 文字
函数: transcribe(wav_bytes) → 文字
"""

import base64
import sys
import threading

from src.core.config import get_config


def _paraformer_transcribe(wav_bytes: bytes) -> str:
    """paraformer-realtime-v2 (流式同步封装)."""
    cfg = get_config()
    import dashscope
    from dashscope.audio.asr import (
        Recognition,
        RecognitionCallbacks,
        RecognitionResult,
    )

    dashscope.api_key = cfg.dashscope_api_key

    sentences: list[str] = []
    done = threading.Event()
    error: list[str | None] = [None]

    class CB(RecognitionCallbacks):
        def on_event(self, result: RecognitionResult):
            try:
                if getattr(result, "is_sentence", False):
                    text = result.get_sentence()
                    if text:
                        sentences.append(text)
            except Exception:
                pass

        def on_complete(self):
            done.set()

        def on_error(self, result):
            error[0] = str(result)
            done.set()

        def on_close(self):
            done.set()

        def on_open(self):
            pass

    recognition = Recognition(
        model="paraformer-realtime-v2",
        format="wav",
        sample_rate=16000,
        callback=CB(),
    )
    recognition.start()
    # 发送完整音频 (一次)
    if hasattr(recognition, "stream_send_audio"):
        recognition.stream_send_audio(wav_bytes)
    elif hasattr(recognition, "send_audio_frame"):
        recognition.send_audio_frame(wav_bytes)
    recognition.stop()
    done.wait(timeout=15)

    if error[0]:
        raise RuntimeError(f"paraformer: {error[0]}")
    return "".join(sentences).strip()


def _omni_transcribe(wav_bytes: bytes) -> str:
    """qwen-omni-flash fallback (paraformer 失败时)."""
    from openai import OpenAI

    cfg = get_config()
    client = OpenAI(
        api_key=cfg.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    b64 = base64.b64encode(wav_bytes).decode()
    resp = client.chat.completions.create(
        model="qwen3.5-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:audio/wav;base64,{b64}",
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": "请转录这段音频中的人声内容, 只输出原文.",
                    },
                ],
            }
        ],
        modalities=["text"],
    )
    return (resp.choices[0].message.content or "").strip()


def transcribe(wav_bytes: bytes) -> str:
    """wav bytes → 文字. 先试 paraformer (便宜), 失败 fallback omni."""
    try:
        text = _paraformer_transcribe(wav_bytes)
        if text:
            return text
        print("[ASR] paraformer 返回空, fallback omni", file=sys.stderr, flush=True)
    except Exception as e:
        print(
            f"[ASR] paraformer 失败, fallback omni: {e}",
            file=sys.stderr,
            flush=True,
        )
    return _omni_transcribe(wav_bytes)


if __name__ == "__main__":
    # 命令行: python -m src.agent.asr < wav → stdout 文字
    audio = sys.stdin.buffer.read()
    sys.stdout.write(transcribe(audio))
