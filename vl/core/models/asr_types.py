"""ASR 转写专用数据类型"""

from dataclasses import dataclass, field


@dataclass
class QwenSegment:
    """Qwen-ASR 单个转写片段"""
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
