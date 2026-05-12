"""场景数据模型"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scene:
    """视频场景"""
    scene_id: str              # "{video_id}_s{index}"
    video_id: str
    index: int                 # 场景在视频中的序号
    start_time: float          # 开始时间(秒)
    end_time: float            # 结束时间(秒)
    start_frame: int           # 起始帧号
    end_frame: int             # 结束帧号
    keyframe_paths: list[str] = field(default_factory=list)
    transition_type: str = "cut"  # "cut" | "dissolve" | "fade"
    clip_embedding: list[float] | None = None
    vlm_caption: str | None = None
    structured_caption: dict | None = None  # Stage 3 结构化描述 JSON
    content_type: str = "main"  # "opening" | "main" | "ending"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "video_id": self.video_id,
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "keyframe_paths": self.keyframe_paths,
            "transition_type": self.transition_type,
            "clip_embedding": self.clip_embedding,
            "vlm_caption": self.vlm_caption,
            "structured_caption": self.structured_caption,
            "content_type": self.content_type,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def get_normalized_caption(self) -> dict:
        """获取归一化的结构化描述。

        统一处理:
          - actions(list) → main_actions(string)
          - interaction → interactions
        """
        cap = self.structured_caption or {}
        if not cap:
            return cap

        # actions(list) → main_actions(string)
        if "actions" in cap and "main_actions" not in cap:
            actions = cap["actions"]
            if isinstance(actions, list):
                cap["main_actions"] = "\uff1b".join(str(a) for a in actions)

        # interaction → interactions
        if "interaction" in cap and "interactions" not in cap:
            cap["interactions"] = cap["interaction"]

        return cap
