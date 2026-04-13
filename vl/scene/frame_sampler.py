"""关键帧提取 - 为每个场景提取代表性关键帧"""

import os

import cv2
from tqdm import tqdm

from vl.core.models.scene import Scene


class FrameSampler:
    """场景关键帧采样器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def sample_keyframes(
        self,
        video_path: str,
        scenes: list[Scene],
        samples_per_scene: int = 1,
    ) -> list[Scene]:
        """
        为每个场景提取关键帧。默认提取场景中间帧。

        Args:
            video_path: 视频文件路径
            scenes: 场景列表
            samples_per_scene: 每个场景提取的帧数

        Returns:
            更新了 keyframe_paths 的场景列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        for scene in tqdm(scenes, desc="提取关键帧"):
            # 清除旧的关键帧路径
            scene.keyframe_paths = []

            if samples_per_scene == 1:
                # 取场景中间帧
                mid_time = (scene.start_time + scene.end_time) / 2
                mid_frame = int(mid_time * fps)
                frame = self._read_frame(cap, mid_frame)
                if frame is not None:
                    path = os.path.join(
                        self.output_dir, f"scene_{scene.index:04d}.jpg"
                    )
                    # cv2.imwrite 不支持中文路径，用 imencode + 写文件
                    encoded = cv2.imencode(".jpg", frame)
                    if encoded[0]:
                        encoded[1].tofile(path)
                        scene.keyframe_paths.append(path.replace("\\", "/"))
            else:
                # 均匀采样
                duration = scene.end_time - scene.start_time
                for j in range(samples_per_scene):
                    t = scene.start_time + duration * (j + 1) / (samples_per_scene + 1)
                    frame_num = int(t * fps)
                    frame = self._read_frame(cap, frame_num)
                    if frame is not None:
                        path = os.path.join(
                            self.output_dir,
                            f"scene_{scene.index:04d}_{j:02d}.jpg",
                        )
                        encoded = cv2.imencode(".jpg", frame)
                        if encoded[0]:
                            encoded[1].tofile(path)
                            scene.keyframe_paths.append(path.replace("\\", "/"))

        cap.release()
        return scenes

    def _read_frame(self, cap: cv2.VideoCapture, frame_num: int):
        """读取指定帧"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        return frame if ret else None
