"""场景检测模块"""

from src.scene.transnet_detector import TransNetDetector


def create_detector(config) -> "TransNetDetector":
    """创建场景检测器 (回退路径使用, 主路径走 speaker_anchor)

    Args:
        config: AppConfig 实例

    Returns:
        TransNetDetector
    """
    return TransNetDetector(
        model_path=config.transnet_model_path,
        threshold=config.transnet_threshold,
        min_scene_len=config.min_scene_len,
    )
