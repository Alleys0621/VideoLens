"""场景检测 - 使用 PySceneDetect 检测视频场景边界"""

import os
from typing import Optional

import cv2
from scenedetect import detect, ContentDetector, AdaptiveDetector, SceneManager, open_video
from vl.core.models.scene import Scene


class SceneDetector:
    """视频场景边界检测器"""

    def __init__(
        self,
        content_threshold: float = 27.0,
        min_scene_len: float = 1.0,
    ):
        self.content_threshold = content_threshold
        self.min_scene_len = min_scene_len

    def detect_scenes(self, video_path: str) -> list[Scene]:
        """
        检测视频中的场景边界，返回 Scene 列表。

        使用 PySceneDetect 的 ContentDetector 检测镜头切换，
        AdaptiveDetector 检测渐变过渡。
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 打开视频获取元数据
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if fps <= 0:
            fps = 25.0

        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # 使用 SceneManager 检测场景
        video = open_video(video_path)
        scene_manager = SceneManager()

        # 添加检测器
        min_frames = int(self.min_scene_len * fps)
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.content_threshold,
                min_scene_len=min_frames,
            )
        )

        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        # 如果没有检测到场景，整个视频作为一个场景
        if not scene_list:
            scene_list = [(video.position_to_timecode(0), video.position_to_timecode(duration))]

        scenes = []
        for i, (start_tc, end_tc) in enumerate(scene_list):
            scene = Scene(
                scene_id=f"{video_id}_s{i}",
                video_id=video_id,
                index=i,
                start_time=start_tc.get_seconds(),
                end_time=end_tc.get_seconds(),
                start_frame=start_tc.get_frames(),
                end_frame=end_tc.get_frames(),
                transition_type="cut",
            )
            scenes.append(scene)

        return scenes
