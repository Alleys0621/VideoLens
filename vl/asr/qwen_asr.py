"""Qwen3-ASR-Flash 语音转写模块"""

import base64
import os
import uuid

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from openai import OpenAI

from vl.core.models.asr_types import QwenSegment
from vl.core.helpers.scene_utils import assign_segments_to_scenes
from vl.core.logging import get_logger

logger = get_logger()


class QwenASR:
    """基于 qwen3-asr-flash 的语音转写器"""

    MAX_CHUNK_BYTES = 8 * 1024 * 1024  # 8MB
    SILENCE_MIN_LEN = 500
    SILENCE_THRESH = -40

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen3-asr-flash",
        language: str = "zh",
        chunk_duration: int = 240,
        silence_min_len: int = 500,
        silence_thresh: int = -40,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.chunk_duration = chunk_duration
        self.silence_min_len = silence_min_len
        self.silence_thresh = silence_thresh

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def transcribe(self, audio_path: str) -> list[QwenSegment]:
        """转写音频文件。"""
        logger.info(f"加载音频: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        total_ms = len(audio)
        logger.info(f"音频时长: {total_ms / 1000:.1f}秒")

        chunks = self._split_audio(audio)
        logger.info(f"音频切分为 {len(chunks)} 个片段")

        all_segments = []
        for i, (chunk_start_ms, chunk_audio) in enumerate(chunks):
            chunk_start_s = chunk_start_ms / 1000
            chunk_end_s = (chunk_start_ms + len(chunk_audio)) / 1000
            logger.info(f"转写片段 {i + 1}/{len(chunks)} "
                        f"[{chunk_start_s:.1f}s - {chunk_end_s:.1f}s]")

            tmp_path = os.path.join(
                os.path.dirname(audio_path),
                f"_chunk_{i}.wav",
            )
            chunk_audio.export(tmp_path, format="wav")

            try:
                segments = self._transcribe_chunk(tmp_path, chunk_start_s, chunk_end_s)
                all_segments.extend(segments)
                logger.info(f"  片段 {i + 1}: {len(segments)} 个转写段")
            except Exception as e:
                logger.warning(f"  片段 {i + 1} 转写失败: {e}")
            finally:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)

        logger.info(f"转写完成: 共 {len(all_segments)} 个片段")
        return all_segments

    def _split_audio(self, audio: AudioSegment) -> list[tuple[int, AudioSegment]]:
        """基于静音检测的音频切块。"""
        max_chunk_ms = self.chunk_duration * 1000

        nonsilent = detect_nonsilent(
            audio,
            min_silence_len=self.silence_min_len,
            silence_thresh=self.silence_thresh,
        )

        if not nonsilent:
            logger.warning("未检测到语音内容")
            return []

        chunks = []
        current_start = nonsilent[0][0]
        current_end = nonsilent[0][1]

        for start, end in nonsilent[1:]:
            if end - current_start <= max_chunk_ms:
                current_end = end
            else:
                chunk_audio = audio[current_start:current_end]
                chunks.append((current_start, chunk_audio))
                current_start = start
                current_end = end

        chunk_audio = audio[current_start:current_end]
        chunks.append((current_start, chunk_audio))

        return chunks

    def _transcribe_chunk(
        self, chunk_path: str, time_offset_start: float, time_offset_end: float
    ) -> list[QwenSegment]:
        """转写单个音频切片 (base64 编码传入)"""
        file_size = os.path.getsize(chunk_path)
        if file_size > self.MAX_CHUNK_BYTES:
            logger.warning(f"音频切片过大 ({file_size / 1024 / 1024:.1f}MB)，跳过")
            return []

        with open(chunk_path, "rb") as f:
            audio_bytes = f.read()
        b64_str = base64.b64encode(audio_bytes).decode()
        data_uri = f"data:audio/wav;base64,{b64_str}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_uri},
                        }
                    ],
                },
            ],
            extra_body={
                "asr_options": {
                    "language": self.language,
                    "enable_itn": True,
                }
            },
        )

        raw = response.choices[0].message.content or ""

        global_emotion = ""
        global_language = ""
        annotations = response.choices[0].message.annotations or []
        for ann in annotations:
            if hasattr(ann, "type") and ann.type == "audio_info":
                global_emotion = getattr(ann, "emotion", "")
                global_language = getattr(ann, "language", "")

        if not raw.strip():
            return []

        return [QwenSegment(
            segment_id=uuid.uuid4().hex[:8],
            text=raw.strip(),
            start_time=time_offset_start,
            end_time=time_offset_end,
            emotion=global_emotion,
            language=global_language,
        )]

    def assign_to_scenes(
        self, segments: list[QwenSegment], scenes: list
    ) -> list[QwenSegment]:
        """将转写片段分配到对应的视频场景"""
        return assign_segments_to_scenes(segments, scenes)
