"""
Stage 2: 视觉处理

流程:
  1. PySceneDetect 场景检测
  2. 关键帧提取 + 字幕帧过滤
  3. Qwen-VL-Max OCR 字幕识别
  4. Qwen-VL-Max 视觉描述生成
  → scenes.json + keyframes/ + visual.json
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np

from src.core.helpers.json_utils import save_json
from src.core.helpers.prompt_loader import load_prompt
from src.core.config import get_config
from src.core.logging import get_logger
from src.scene.detector import SceneDetector

logger = get_logger()


def extract_keyframes(video_path, scenes, output_dir, samples_per_scene=8):
    """为每个场景提取关键帧"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    for scene in scenes:
        start_frame = scene["start_frame"]
        end_frame = scene["end_frame"]
        scene_idx = scene["index"]
        n_frames = end_frame - start_frame

        if n_frames <= 0:
            continue

        # 均匀采样
        if n_frames <= samples_per_scene:
            indices = list(range(start_frame, end_frame))
        else:
            indices = np.linspace(start_frame, end_frame - 1, samples_per_scene, dtype=int)

        for j, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(keyframes_dir, f"scene_{scene_idx:04d}_frame_{j:02d}.jpg")
                cv2.imwrite(path, frame)
                scene.setdefault("keyframe_paths", []).append(path)

    cap.release()
    return scenes


def filter_subtitle_frames(scenes, audio_segments=None):
    """根据 ASR 时间戳过滤关键帧: 字幕帧只保留中心帧"""
    for scene in scenes:
        kf = scene.get("keyframe_paths", [])
        if len(kf) <= 2:
            continue

        # 简单策略: 保留首帧 + 末帧 + 中心帧
        mid = len(kf) // 2
        kept = [kf[0], kf[mid], kf[-1]] if len(kf) > 2 else kf

        # 删除未保留的帧文件
        for p in kf:
            if p not in kept and os.path.isfile(p):
                os.remove(p)

        scene["keyframe_paths"] = kept

    return scenes


def run_ocr(scenes, config):
    """对有字幕的场景进行 OCR"""
    from src.core.llm.qwen_vl import QwenVLClient

    vl_client = QwenVLClient()
    user_tpl, _ = load_prompt(config, "stage2_scene_caption")

    ocr_results = {}
    for scene in scenes:
        kf = scene.get("keyframe_paths", [])
        if not kf:
            continue

        # 用第一帧做简单 OCR
        first_kf = kf[0]
        if not os.path.isfile(first_kf):
            continue

        try:
            result = vl_client.analyze_image(
                first_kf,
                "请提取这张图片中的所有文字，只输出文字内容，不要解释。",
                stage="stage2_ocr",
            )
            if result:
                ocr_results[scene["index"]] = result
        except Exception as e:
            logger.warning(f"OCR 场景 {scene['index']} 失败: {e}")

    return ocr_results


def run_captions(scenes, audio_result, config):
    """为每个场景生成视觉描述"""
    from src.core.llm.qwen_vl import QwenVLClient

    vl_client = QwenVLClient()
    user_tpl, _ = load_prompt(config, "stage2_scene_caption")

    # 从 audio_result 构建台词索引
    audio_segs = audio_result.get("segments", []) if audio_result else []

    captions = {}
    for scene in scenes:
        kf = scene.get("keyframe_paths", [])
        if not kf:
            continue

        # 找该场景时间范围内的台词
        s, e = scene["start_time"], scene["end_time"]
        scene_text = " ".join(
            seg["text"] for seg in audio_segs
            if s <= seg.get("begin_time", 0) <= e or s <= seg.get("end_time", 0) <= e
        )

        first_kf = kf[0]
        if not os.path.isfile(first_kf):
            continue

        prompt = user_tpl.format(
            audiotext=scene_text[:200] if scene_text else "无",
            caption="",
        )

        try:
            result = vl_client.analyze_image(
                first_kf,
                prompt if prompt else "请简要描述这个画面的内容。",
                stage="stage2_caption",
            )
            if result:
                captions[scene["index"]] = result
        except Exception as e:
            logger.warning(f"Caption 场景 {scene['index']} 失败: {e}")

    return captions


def run_stage2(video_dir: str, output_dir: str, audio_result: dict = None) -> dict:
    """Stage 2: 视觉处理

    Args:
        video_dir: 视频目录名
        output_dir: 输出目录
        audio_result: Stage 1 的输出 (可选)

    Returns:
        visual.json 数据 (dict)
    """
    config = get_config()
    video_path = os.path.join("data", "videos", f"{video_dir}.mp4")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    print("=" * 60)
    print("Stage 2: 视觉处理")
    print(f"视频: {video_dir}")
    print("=" * 60)

    # Step 2a: 场景检测
    print("[Stage 2a] 场景检测...")
    detector = SceneDetector(
        content_threshold=config.content_threshold,
        min_scene_len=config.min_scene_len,
    )
    scene_objs = detector.detect_scenes(video_path)
    scenes = [s.to_dict() for s in scene_objs]
    print(f"  检测到 {len(scenes)} 个场景")

    # Step 2b: 关键帧提取
    print("[Stage 2b] 关键帧提取...")
    scenes = extract_keyframes(video_path, scenes, output_dir)
    total_kf = sum(len(s.get("keyframe_paths", [])) for s in scenes)
    print(f"  提取 {total_kf} 个关键帧")

    # Step 2c: 帧过滤
    print("[Stage 2c] 帧过滤...")
    scenes = filter_subtitle_frames(scenes, audio_result)

    # 保存场景数据
    save_json(scenes, os.path.join(output_dir, "scenes.json"))

    # Step 2d: OCR (可选)
    print("[Stage 2d] OCR 字幕识别...")
    ocr_results = run_ocr(scenes, config)
    print(f"  OCR 完成: {len(ocr_results)} 个场景有字幕")

    # Step 2e: 视觉描述
    print("[Stage 2e] 视觉描述生成...")
    captions = run_captions(scenes, audio_result, config)
    print(f"  Caption 完成: {len(captions)} 个场景有描述")

    # 合并输出
    visual_data = {
        "scenes": scenes,
        "ocr": ocr_results,
        "captions": captions,
    }
    save_json(visual_data, os.path.join(output_dir, "visual.json"))

    print(f"\n{'=' * 60}")
    print(f"结果: {os.path.join(output_dir, 'visual.json')}")
    print(f"场景数: {len(scenes)}, OCR: {len(ocr_results)}, Caption: {len(captions)}")
    print(f"{'=' * 60}")

    return visual_data
