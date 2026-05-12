"""Qwen-Omni 全模态理解数据模型"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OmniSegment:
    """单个转写片段 (含说话人 + 情感)"""
    segment_id: str
    text: str
    start_time: float
    end_time: float
    speaker: str = ""
    emotion: str = ""
    language: str = ""
    scene_id: str = ""
    confidence: float = 1.0

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
        }


@dataclass
class OmniSceneDescription:
    """片段的视觉场景描述"""
    time_of_day: str = ""
    space: str = ""
    subspace: str = ""
    scene: str = ""
    characters: list = field(default_factory=list)
    main_actions: str = ""
    interactions: str = ""
    emotion: str = ""
    plot_state: str = ""

    def to_dict(self):
        return {
            "time_of_day": self.time_of_day,
            "space": self.space,
            "subspace": self.subspace,
            "scene": self.scene,
            "characters": self.characters,
            "main_actions": self.main_actions,
            "interactions": self.interactions,
            "emotion": self.emotion,
            "plot_state": self.plot_state,
        }


@dataclass
class OmniChunkResult:
    """单个片段的完整理解结果"""
    chunk_index: int
    time_start: float
    time_end: float
    segments: list[OmniSegment] = field(default_factory=list)
    speakers: list[dict] = field(default_factory=list)
    scene_description: Optional[OmniSceneDescription] = None
    raw_text: str = ""
    content_type: str = "main"  # "opening" | "main" | "ending"
