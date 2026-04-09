"""FFmpeg 音频提取 (移植自 MoocAI)"""

import os
import subprocess


def extract_audio(
    video_path: str,
    output_audio: str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """从视频中提取音频为 WAV"""
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_audio,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="ignore",
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 提取音频失败:\n{result.stderr}")

    return output_audio
