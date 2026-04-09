"""视频元数据模型"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoMeta:
    """视频处理元数据"""
    video_id: str
    file_path: str
    title: str = ""
    genre: str = "movie"      # "movie" | "tv" | "anime"
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    scene_count: int = 0
    processing_status: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "file_path": self.file_path,
            "title": self.title,
            "genre": self.genre,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "scene_count": self.scene_count,
            "processing_status": self.processing_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMeta":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
