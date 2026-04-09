"""Stage 1: 视频导入 + 场景分割 + 关键帧提取"""

from vl_core.config import AppConfig
from vl_core.models.scene import Scene
from vl_scene.detector import SceneDetector
from vl_scene.frame_sampler import FrameSampler

from vl_core.logging import get_logger

logger = get_logger()


def run_stage1(
    video_path: str,
    output_dir: str,
    config: AppConfig,
) -> list[Scene]:
    """
    执行 Stage 1: 场景检测和关键帧提取。

    Args:
        video_path: 视频文件路径
        output_dir: 场景输出目录
        config: 应用配置

    Returns:
        检测到的场景列表
    """
    logger.info("[Stage 1] 开始场景分割...")

    # 1. 场景检测
    detector = SceneDetector(
        content_threshold=config.content_threshold,
        min_scene_len=config.min_scene_len,
    )
    scenes = detector.detect_scenes(video_path)
    logger.info(f"检测到 {len(scenes)} 个场景")

    # 2. 关键帧提取
    keyframes_dir = f"{output_dir}/keyframes"
    sampler = FrameSampler(keyframes_dir)
    scenes = sampler.sample_keyframes(video_path, scenes, samples_per_scene=1)
    logger.info(f"关键帧提取完成，保存到 {keyframes_dir}")

    return scenes
