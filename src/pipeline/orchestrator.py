"""
VideoLens Pipeline 编排器 — 3-Stage 统一 Pipeline

Stage 1: 音频处理 (ASR + 声纹 + 情感)
Stage 2: 视觉处理 (场景检测 + OCR + Caption)
Stage 3: 结构化知识库
"""

import os
import sys

from src.core.logging import get_logger
from src.core.path_utils import (
    get_show_name,
    load_voiceprint_config,
)

sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger()


def run_pipeline(video_dir: str, stage: int = 0, skip_theme: bool = False, chunk_dur: int = 60, vp_threshold: float = 0.4):
    """运行 Pipeline

    Args:
        video_dir: 视频目录名, 如 "052 鸟蛋之争" 或 "家有儿女/第001集"
        stage: 运行到哪个 stage (0=全部, 1/2/3)
        skip_theme: 是否跳过片头/片尾曲检测
        chunk_dur: Omni chunk 时长
        vp_threshold: 声纹置信度阈值, 低于此值统一标记为 '路人' (默认 0.4, 0=不过滤)
    """
    output_dir = os.path.join("data", "output", video_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 解析声纹配置
    show_name = get_show_name(video_dir)
    group_id, name_map = load_voiceprint_config(show_name)
    if show_name:
        logger.info(f"影视作品: {show_name}, 声纹组: {group_id or '无'}")

    audio_result = None
    visual_result = None

    # Stage 1: 音频
    if stage == 0 or stage == 1:
        from src.pipeline.stage1_audio import run_stage1
        audio_result = run_stage1(
            video_dir, output_dir,
            skip_theme=skip_theme, chunk_dur=chunk_dur, vp_threshold=vp_threshold,
            group_id=group_id, name_map=name_map,
        )

        if stage == 1:
            return

    # Stage 2: 视觉
    if stage == 0 or stage == 2:
        from src.pipeline.stage2_visual import run_stage2
        visual_result = run_stage2(video_dir, output_dir, audio_result=audio_result)

        if stage == 2:
            return

    # Stage 3: 知识库
    if stage == 0 or stage == 3:
        from src.pipeline.stage3 import run_stage3
        run_stage3(video_dir, output_dir)
