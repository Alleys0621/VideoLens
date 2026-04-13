"""角色数据模型"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FaceDetection:
    """人脸检测结果"""
    image_path: str
    bbox: list[float]       # [x1, y1, x2, y2]
    embedding: list[float]  # 512-dim insightface embedding
    confidence: float
    scene_id: str = ""


@dataclass
class Character:
    """视频中的角色"""
    character_id: str          # "char_01"
    label: str                 # "Character_01" 或推断的名字
    representative_face_path: str | None = None
    appearance_scenes: list[str] = field(default_factory=list)
    is_anime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "label": self.label,
            "representative_face_path": self.representative_face_path,
            "appearance_scenes": self.appearance_scenes,
            "is_anime": self.is_anime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
