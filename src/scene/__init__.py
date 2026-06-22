"""场景检测模块"""

from src.scene.detector import SceneDetector


def create_detector(config) -> "SceneDetector | TransNetDetector":
    """根据配置创建场景检测器

    Args:
        config: AppConfig 实例

    Returns:
        SceneDetector 或 TransNetDetector
    """
    detector_type = getattr(config, "scene_detector", "pyscenedetect")

    if detector_type == "transnetv2":
        from src.scene.transnet_detector import TransNetDetector
        return TransNetDetector(
            model_path=config.transnet_model_path,
            threshold=config.transnet_threshold,
            min_scene_len=config.min_scene_len,
        )

    return SceneDetector(
        content_threshold=config.content_threshold,
        min_scene_len=config.min_scene_len,
    )
