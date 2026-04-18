"""Qwen3-ASR-Flash 语音转写模块

通过 DashScope OpenAI 兼容 API 调用 qwen3-asr-flash 模型，
使用 base64 编码传入本地音频。

qwen3-asr-flash 为 ASR 专用模型：
  - 不支持 system message
  - 返回整段音频的纯文本（无逐句时间戳/说话人）
  - 通过 annotations 返回全局 emotion 和 language
  - 单次最大 10MB base64 数据

流程:
  1. pydub 静音检测 → 音频切块 (确保每块 < 10MB)
  2. 每个切片 base64 编码，调用 qwen3-asr-flash
  3. 合并结果并加上全局时间偏移
"""

import base64
import os
import uuid
from dataclasses import dataclass, field

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from openai import OpenAI

from vl.core.logging import get_logger

logger = get_logger()


@dataclass
class QwenSegment:
    """单个转写片段"""
    segment_id: str
    text: str
    start_time: float  # 全局秒
    end_time: float
    speaker: str = ""
    emotion: str = ""
    language: str = ""
    scene_id: str = ""
    confidence: float = 1.0
    words: list = field(default_factory=list)

    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "language": self.language,
            "scene_id": self.scene_id,
            "confidence": self.confidence,
            "words": self.words,
        }


class QwenASR:
    """基于 qwen3-asr-flash 的语音转写器"""

    # qwen3-asr-flash base64 编码后最大 10MB
    MAX_CHUNK_BYTES = 8 * 1024 * 1024  # 8MB 原始文件，base64 后约 10.7MB
    # 静音检测参数
    SILENCE_MIN_LEN = 500    # 静音最短时长 ms
    SILENCE_THRESH = -40     # 静音阈值 dBFS

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
        """
        转写音频文件。

        1. 静音检测切块
        2. 逐块调用 API
        3. 合并结果
        """
        logger.info(f"加载音频: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        total_ms = len(audio)
        logger.info(f"音频时长: {total_ms / 1000:.1f}秒")

        # 1. 静音切块
        chunks = self._split_audio(audio)
        logger.info(f"音频切分为 {len(chunks)} 个片段")

        # 2. 逐块转写
        all_segments = []
        for i, (chunk_start_ms, chunk_audio) in enumerate(chunks):
            chunk_start_s = chunk_start_ms / 1000
            chunk_end_s = (chunk_start_ms + len(chunk_audio)) / 1000
            logger.info(f"转写片段 {i + 1}/{len(chunks)} "
                        f"[{chunk_start_s:.1f}s - {chunk_end_s:.1f}s]")

            # 导出临时 wav
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
        """
        基于静音检测的音频切块。

        策略:
        - 检测非静音区间
        - 将连续的非静音区间合并，直到总时长超过 chunk_duration
        - 在最近的静音边界切分
        """
        max_chunk_ms = self.chunk_duration * 1000

        # 检测非静音区间
        nonsilent = detect_nonsilent(
            audio,
            min_silence_len=self.silence_min_len,
            silence_thresh=self.silence_thresh,
        )

        if not nonsilent:
            logger.warning("未检测到语音内容")
            return []

        # 合并为 chunks
        chunks = []
        current_start = nonsilent[0][0]
        current_end = nonsilent[0][1]

        for start, end in nonsilent[1:]:
            # 如果加入这段不超过最大时长，合并
            if end - current_start <= max_chunk_ms:
                current_end = end
            else:
                # 超出 → 切出当前 chunk
                chunk_audio = audio[current_start:current_end]
                chunks.append((current_start, chunk_audio))
                current_start = start
                current_end = end

        # 最后一段
        chunk_audio = audio[current_start:current_end]
        chunks.append((current_start, chunk_audio))

        return chunks

    def _transcribe_chunk(
        self, chunk_path: str, time_offset_start: float, time_offset_end: float
    ) -> list[QwenSegment]:
        """转写单个音频切片 (base64 编码传入)"""
        # 检查文件大小
        file_size = os.path.getsize(chunk_path)
        if file_size > self.MAX_CHUNK_BYTES:
            logger.warning(f"音频切片过大 ({file_size / 1024 / 1024:.1f}MB)，跳过")
            return []

        # 读取并编码为 base64 Data URL
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

        # 提取全局 emotion/language
        global_emotion = ""
        global_language = ""
        annotations = response.choices[0].message.annotations or []
        for ann in annotations:
            if hasattr(ann, "type") and ann.type == "audio_info":
                global_emotion = getattr(ann, "emotion", "")
                global_language = getattr(ann, "language", "")

        if not raw.strip():
            return []

        # qwen3-asr-flash 返回整段文本，作为一个 segment
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
        for seg in segments:
            mid = (seg.start_time + seg.end_time) / 2
            for scene in scenes:
                if scene.start_time <= mid <= scene.end_time:
                    seg.scene_id = scene.scene_id
                    break
        return segments
