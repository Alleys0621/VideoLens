"""角色数据模型"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Character:
    """视频中的角色"""
    character_id: str          # "char_01"
    label: str                 # 角色名
    appearance_scenes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "label": self.label,
            "appearance_scenes": self.appearance_scenes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
