"""语音转录 - 使用 faster-whisper 将音频转为带时间戳的文本"""

import os
import uuid
from typing import Optional

from faster_whisper import WhisperModel

from vl.core.models.transcript import TranscriptSegment, Word
from vl.core.logging import get_logger

logger = get_logger()


def _ensure_hf_env():
    """确保 HuggingFace 环境变量已设置（国内镜像等）"""
    from dotenv import load_dotenv
    from pathlib import Path

    # 找到项目根目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            try:
                content = (parent / "pyproject.toml").read_text(encoding="utf-8")
                if 'name = "videolens"' in content:
                    env_path = parent / ".env"
                    if env_path.is_file():
                        load_dotenv(str(env_path), override=False)
                    break
            except (OSError, UnicodeDecodeError):
                pass


class Transcriber:
    """Whisper 语音转录器"""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        # 确保环境变量在模型下载前设置
        _ensure_hf_env()

        logger.info(f"加载 ASR 模型: {model_size} (device={device}, compute_type={compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("ASR 模型加载完成")

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """
        转录音频文件，返回带时间戳的片段列表。
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info(f"开始转录: {audio_path}")

        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        logger.info(
            "音频信息: 时长=%.1f秒, 语言=%s",
            info.duration,
            info.language,
        )

        results = []
        for seg in segments:
            words = [
                Word(text=w.word, start=w.start, end=w.end, confidence=w.probability)
                for w in (seg.words or [])
            ]
            results.append(TranscriptSegment(
                segment_id=str(uuid.uuid4())[:8],
                scene_id="",
                speaker_id="SPEAKER_00",
                text=seg.text.strip(),
                start_time=seg.start,
                end_time=seg.end,
                words=words,
                confidence=seg.avg_logprob if seg.avg_logprob else 0.0,
            ))

        logger.info(f"转录完成: 共 {len(results)} 个片段，总时长 {info.duration:.1f}秒")
        return results

    def assign_to_scenes(
        self,
        segments: list[TranscriptSegment],
        scenes: list,
    ) -> list[TranscriptSegment]:
        """将转录片段分配到对应的场景"""
        for seg in segments:
            mid_time = (seg.start_time + seg.end_time) / 2
            for scene in scenes:
                if scene.start_time <= mid_time <= scene.end_time:
                    seg.scene_id = scene.scene_id
                    break
        return segments
