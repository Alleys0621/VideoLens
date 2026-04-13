"""人脸检测 - 使用 insightface 检测图片中的人脸并提取嵌入向量"""

import numpy as np
from PIL import Image

from vl.core.config import get_config
from vl.core.logging import get_logger
from vl.core.models.character import FaceDetection

logger = get_logger()


class FaceDetector:
    """基于 insightface 的人脸检测器"""

    def __init__(self, detection_threshold: float = 0.5):
        logger.info("加载人脸检测模型: insightface buffalo_l")
        import insightface
        from insightface.app import FaceAnalysis

        self.threshold = detection_threshold
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("人脸检测模型加载完成")

    def detect(self, image_path: str, scene_id: str = "") -> list[FaceDetection]:
        """检测单张图片中的人脸"""
        # 使用 PIL 读取避免中文路径问题，再转 BGR 给 insightface
        img = Image.open(image_path).convert("RGB")
        img_bgr = np.array(img)[:, :, ::-1]

        faces = self.app.get(img_bgr)

        results = []
        for face in faces:
            if face.det_score < self.threshold:
                continue
            results.append(FaceDetection(
                image_path=image_path,
                bbox=face.bbox.tolist(),
                embedding=face.embedding.tolist(),
                confidence=float(face.det_score),
                scene_id=scene_id,
            ))

        return results

    def detect_batch(
        self,
        image_paths: list[str],
        scene_ids: list[str],
    ) -> list[FaceDetection]:
        """批量检测多张图片中的人脸"""
        all_detections = []
        for path, sid in zip(image_paths, scene_ids):
            try:
                detections = self.detect(path, scene_id=sid)
                all_detections.extend(detections)
            except Exception as e:
                logger.warning(f"人脸检测失败 ({path}): {e}")
        logger.info(f"人脸检测完成: 共检测到 {len(all_detections)} 张人脸")
        return all_detections
