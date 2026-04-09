"""语音转录 - 使用 faster-whisper 将音频转为带时间戳的文本"""

import os
import uuid
from typing import Optional

from faster_whisper import WhisperModel

from vl_core.models.transcript import TranscriptSegment, Word


class Transcriber:
    """Whisper 语音转录器"""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """
        转录音频文件，返回带时间戳的片段列表。
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
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
