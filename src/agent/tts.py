"""TTS 服务: DashScope CosyVoice 文字→语音.

用 dashscope SDK 的 SpeechSynthesizer (tts_v2).
函数: synthesize(text) → base64 音频字符串
命令行: python -m src.agent.tts "你好" → stdout 输出 base64
"""

import base64
import sys

from src.core.config import get_config


def synthesize(text: str) -> str | None:
    """文字 → base64 编码的音频 (mp3).

    Returns:
        base64 字符串, 失败返回 None
    """
    if not text or not text.strip():
        return None

    cfg = get_config()

    try:
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer

        dashscope.api_key = cfg.dashscope_api_key

        # CosyVoice v2, longxiaochun_v2 是中文女声 (亲切, 适合陪看 Alleys 人设)
        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v2-0.5B",
            voice="longxiaochun_v2",
        )

        audio = synthesizer.call(text)

        # audio 可能是 AudioResult 对象 / bytes / 其他
        if hasattr(audio, "get_audio_data"):
            audio_bytes = audio.get_audio_data()
        elif isinstance(audio, dict) and "output" in audio:
            audio_bytes = audio["output"].get("audio")
        elif isinstance(audio, (bytes, bytearray)):
            audio_bytes = bytes(audio)
        else:
            audio_bytes = getattr(audio, "data", None)

        if audio_bytes:
            return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        print(f"[TTS] synthesize 失败: {e}", file=sys.stderr, flush=True)

    return None


if __name__ == "__main__":
    # 命令行模式: python -m src.agent.tts "你好" → stdout 输出 base64
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    result = synthesize(text)
    if result:
        sys.stdout.write(result)
