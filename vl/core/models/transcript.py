"""台词数据模型"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    """词级别时间戳"""
    text: str
    start: float
    end: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptSegment:
    """带说话人信息的台词片段"""
    segment_id: str
    scene_id: str
    speaker_id: str           # "SPEAKER_00"
    text: str
    start_time: float
    end_time: float
    words: list[Word] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "scene_id": self.scene_id,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "words": [w.to_dict() for w in self.words],
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        words = [Word(**w) for w in data.get("words", [])]
        return cls(
            segment_id=data["segment_id"],
            scene_id=data.get("scene_id", ""),
            speaker_id=data.get("speaker_id", "SPEAKER_00"),
            text=data["text"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            words=words,
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class DiarizationSegment:
    """说话人片段"""
    speaker: str       # "SPEAKER_00"
    start_time: float
    end_time: float
