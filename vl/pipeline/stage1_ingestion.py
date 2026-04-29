"""Stage 1: 视频导入 + 场景分割 + 关键帧提取"""

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.scene.detector import SceneDetector
from vl.scene.frame_sampler import FrameSampler

from vl.core.logging import get_logger

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
    logger.info(
        "场景检测参数: content_threshold=%.2f, min_scene_len=%d",
        config.content_threshold,
        config.min_scene_len,
    )

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
    scenes = sampler.sample_keyframes(video_path, scenes, samples_per_scene=config.samples_per_scene)
    keyframe_count = sum(len(s.keyframe_paths) for s in scenes if s.keyframe_paths)
    logger.info(f"关键帧提取完成: 共提取 {keyframe_count} 张关键帧，保存到 {keyframes_dir}")

    return scenes
