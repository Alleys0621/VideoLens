"""
VideoLens Pipeline 编排器 — 3-Stage 统一 Pipeline

Stage 1: 音频处理 (ASR + 声纹 + 情感)
Stage 2: 视觉处理 (场景检测 + OCR + Caption)
Stage 3: 结构化知识库
"""

import os
import sys

import yaml

sys.stdout.reconfigure(encoding="utf-8")


def resolve_video_path(video_dir: str) -> str:
    """解析视频文件路径, 支持 data/videos/ 下的子目录结构

    例如:
      "052 鸟蛋之争"        → data/videos/喜羊羊与灰太狼/052 鸟蛋之争.mp4
      "家有儿女/第001集"     → data/videos/家有儿女/第001集.mp4
    """
    # 直接路径
    direct = os.path.join("data", "videos", f"{video_dir}.mp4")
    if os.path.isfile(direct):
        return direct

    # 在子目录中搜索
    videos_root = "data/videos"
    if os.path.isdir(videos_root):
        for subdir in os.listdir(videos_root):
            subdir_path = os.path.join(videos_root, subdir)
            if os.path.isdir(subdir_path):
                candidate = os.path.join(subdir_path, f"{video_dir}.mp4")
                if os.path.isfile(candidate):
                    return candidate

    # 返回默认路径 (后续会报错)
    return direct


def get_show_name(video_dir: str) -> str:
    """从 video_dir 推断所属影视作品名

    Returns:
        如 "喜羊羊与灰太狼", "家有儿女", 或 ""
    """
    videos_root = "data/videos"

    # 子目录格式: "家有儿女/第001集"
    if "/" in video_dir or "\\" in video_dir:
        parts = video_dir.replace("\\", "/").split("/")
        return parts[0] if parts else ""

    # 平铺格式: 搜索哪个子目录包含此文件
    if os.path.isdir(videos_root):
        for subdir in os.listdir(videos_root):
            subdir_path = os.path.join(videos_root, subdir)
            if os.path.isdir(subdir_path):
                candidate = os.path.join(subdir_path, f"{video_dir}.mp4")
                if os.path.isfile(candidate):
                    return subdir

    return ""


def load_voiceprint_config(show_name: str):
    """从 pipeline.yaml 加载对应影视作品的声纹配置

    Returns:
        (group_id, name_map) 或 ("", None) 如果未配置
    """
    config_path = os.path.join("config", "pipeline.yaml")
    if not os.path.isfile(config_path):
        return "", None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    groups = cfg.get("voiceprint_groups", {})
    show_cfg = groups.get(show_name, {})
    if not show_cfg:
        return "", None

    return show_cfg.get("group_id", ""), show_cfg.get("name_mapping", {})


def run_pipeline(video_dir: str, stage: int = 0, skip_theme: bool = False, chunk_dur: int = 60, vp_threshold: float = 0.0):
    """运行 Pipeline

    Args:
        video_dir: 视频目录名, 如 "052 鸟蛋之争" 或 "家有儿女/第001集"
        stage: 运行到哪个 stage (0=全部, 1/2/3)
        skip_theme: 是否跳过片头/片尾曲检测
        chunk_dur: Omni chunk 时长
        vp_threshold: 声纹置信度阈值, 低于此值标为 '路人' (0=不过滤)
    """
    output_dir = os.path.join("data", "output", video_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 解析声纹配置
    show_name = get_show_name(video_dir)
    group_id, name_map = load_voiceprint_config(show_name)
    if show_name:
        print(f"影视作品: {show_name}, 声纹组: {group_id or '无'}")

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
        from src.pipeline.stage3_knowledge import run_stage3
        run_stage3(output_dir, audio_result=audio_result, visual_result=visual_result)
