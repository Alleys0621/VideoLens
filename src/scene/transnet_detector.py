"""TransNetV2 场景检测 - 基于 ONNX Runtime 的轻量推理"""

import os

import cv2
import numpy as np

from src.core.models.scene import Scene

WINDOW_SIZE = 100
STEP_SIZE = 50
PAD_START = 25
INPUT_SIZE = (27, 48, 3)  # (H, W, C)


class TransNetDetector:
    """基于 TransNetV2 ONNX 模型的场景边界检测器

    与 SceneDetector 接口一致: detect_scenes(video_path) -> list[Scene]
    推理完成后调用 release() 释放 ONNX session。
    """

    def __init__(
        self,
        model_path: str = "models/transnetv2.onnx",
        threshold: float = 0.5,
        min_scene_len: float = 1.0,
    ):
        import onnxruntime

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TransNetV2 模型文件不存在: {model_path}\n"
                f"请先运行: python scripts/export_transnetv2_onnx.py"
            )

        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def detect_scenes(self, video_path: str) -> list[Scene]:
        """检测视频场景边界"""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if fps <= 0:
            fps = 25.0

        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # 读取并预处理所有帧
        frames = self._read_frames(video_path, total_frames)
        if len(frames) == 0:
            return [self._make_scene(video_id, 0, 0, total_frames - 1, fps)]

        # 滑窗推理
        predictions = self._predict(frames)
        cap.release()

        # 转场景边界
        scene_pairs = self._predictions_to_scenes(predictions, self.threshold)

        # 过滤过短场景
        min_frames = int(self.min_scene_len * fps)
        scene_pairs = self._filter_short_scenes(scene_pairs, min_frames)

        if len(scene_pairs) == 0:
            scene_pairs = [[0, len(predictions) - 1]]

        scenes = []
        for i, (start, end) in enumerate(scene_pairs):
            scenes.append(self._make_scene(video_id, i, start, end, fps))

        return scenes

    def release(self):
        """释放 ONNX session，释放内存"""
        self._session = None

    # ══════════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════════

    def _read_frames(self, video_path: str, total_frames: int) -> np.ndarray:
        """读取视频所有帧, resize 到 48x27, 返回 [N, 27, 48, 3] uint8"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (48, 27))  # (W, H) → (48, 27)
            frames.append(frame)
        cap.release()

        if not frames:
            return np.array([])
        return np.stack(frames).astype(np.uint8)  # [N, 27, 48, 3]

    def _predict(self, frames: np.ndarray) -> np.ndarray:
        """滑窗推理, 返回每帧的 transition probability [N]"""
        n = len(frames)

        # 首尾 padding
        start_frame = frames[0:1]
        end_frame = frames[-1:]
        pad_end = STEP_SIZE - (n % STEP_SIZE if n % STEP_SIZE != 0 else STEP_SIZE)

        padded = np.concatenate(
            [start_frame] * PAD_START
            + [frames]
            + [end_frame] * (PAD_START + pad_end),
            axis=0,
        )

        predictions = []
        ptr = 0
        while ptr + WINDOW_SIZE <= len(padded):
            batch = padded[ptr : ptr + WINDOW_SIZE]
            batch_input = batch[np.newaxis]  # [1, 100, 27, 48, 3]

            pred = self._session.run(
                ["predictions"], {"frames": batch_input}
            )[0]  # [1, 100, 1]
            predictions.append(pred[0])  # [100, 1]

            ptr += STEP_SIZE

        # 合并重叠窗口: 每个窗口取前 STEP_SIZE 帧 (与 PyTorch 版一致)
        predictions = np.concatenate(predictions, axis=0)[:n]
        return predictions.squeeze(-1)  # [N]

    @staticmethod
    def _predictions_to_scenes(
        predictions: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """将逐帧概率转为 [start, end] 场景对"""
        binary = (predictions > threshold).astype(np.uint8)

        scenes = []
        t_prev, start = 0, 0
        for i, t in enumerate(binary):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t

        if t == 0:
            scenes.append([start, len(binary) - 1])

        if not scenes:
            return np.array([[0, len(binary) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def _filter_short_scenes(
        scenes: np.ndarray, min_frames: int
    ) -> np.ndarray:
        """过滤时长不足 min_frames 的场景"""
        if min_frames <= 0:
            return scenes

        filtered = []
        for start, end in scenes:
            if end - start + 1 >= min_frames:
                filtered.append([start, end])

        if not filtered:
            return scenes  # 全部被过滤则保留原始结果
        return np.array(filtered, dtype=np.int32)

    @staticmethod
    def _make_scene(
        video_id: str, index: int, start_frame: int, end_frame: int, fps: float
    ) -> Scene:
        return Scene(
            scene_id=f"{video_id}_s{index}",
            video_id=video_id,
            index=index,
            start_time=round(start_frame / fps, 3),
            end_time=round(end_frame / fps, 3),
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            transition_type="cut",
        )
