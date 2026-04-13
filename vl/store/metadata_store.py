"""JSON 元数据存储"""

import os
from typing import Any

from vl.core.helpers.json_utils import save_json, load_json


class MetadataStore:
    """基于 JSON 文件的场景元数据存储"""

    def __init__(self, path: str):
        self.path = path

    def save(self, data: Any):
        """保存元数据"""
        save_json(data, self.path, ensure_dir=True)

    def load(self) -> Any:
        """加载元数据"""
        if os.path.isfile(self.path):
            return load_json(self.path)
        return {}

    def get_scene(self, scene_id: str) -> dict | None:
        """获取单个场景的元数据"""
        data = self.load()
        scenes = data.get("scenes", [])
        for scene in scenes:
            if scene.get("scene_id") == scene_id:
                return scene
        return None

    def update_scene(self, scene_id: str, updates: dict):
        """更新单个场景的元数据"""
        data = self.load()
        scenes = data.get("scenes", [])
        for scene in scenes:
            if scene.get("scene_id") == scene_id:
                scene.update(updates)
                break
        self.save(data)
