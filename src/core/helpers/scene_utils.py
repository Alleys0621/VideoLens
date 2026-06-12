"""场景工具 - 场景-片段对齐"""

from typing import Any


def assign_segments_to_scenes(
    segments: list[Any],
    scenes: list[Any],
    time_attr: str = "start_time",
    end_attr: str = "end_time",
    scene_id_attr: str = "scene_id",
):
    """将时间片段分配到对应的视频场景 (基于中点时间匹配)。

    修改 segments 中每个元素的 scene_id_attr 属性。
    """
    for seg in segments:
        mid = (getattr(seg, time_attr) + getattr(seg, end_attr)) / 2
        for scene in scenes:
            if scene.start_time <= mid <= scene.end_time:
                setattr(seg, scene_id_attr, scene.scene_id)
                break
    return segments
