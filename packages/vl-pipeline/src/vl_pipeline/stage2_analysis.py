"""Stage 2: 多维分析 - ASR + CLIP + VLM 场景描述"""

import os
import numpy as np

from vl_core.config import AppConfig
from vl_core.models.scene import Scene
from vl_core.helpers.json_utils import save_json
from vl_asr.transcriber import Transcriber
from vl_vision.clip_encoder import CLIPEncoder

from vl_core.logging import get_logger

logger = get_logger()


def run_stage2(
    video_path: str,
    video_id: str,
    scenes: list[Scene],
    audio_path: str,
    output_dir: str,
    config: AppConfig,
):
    """
    执行 Stage 2: 对每个场景进行多维分析。

    - ASR: 音频转录
    - CLIP: 语义向量编码
    - VLM: 场景描述生成
    """
    # 1. ASR 转录
    logger.info("[Stage 2.1] 开始语音转录...")
    transcriber = Transcriber(
        model_size=config.model_whisper,
        language=config.asr_language,
        beam_size=config.asr_beam_size,
        vad_filter=config.asr_vad_filter,
    )
    segments = transcriber.transcribe(audio_path)
    segments = transcriber.assign_to_scenes(segments, scenes)

    # 保存转录结果
    transcript_dir = os.path.join(output_dir, "transcripts", video_id)
    os.makedirs(transcript_dir, exist_ok=True)
    save_json(
        [s.to_dict() for s in segments],
        os.path.join(transcript_dir, "transcript.json"),
    )
    logger.info(f"转录完成: {len(segments)} 个片段")

    # 构建场景到台词的映射
    scene_transcripts: dict[str, list[str]] = {}
    for seg in segments:
        scene_transcripts.setdefault(seg.scene_id, []).append(seg.text)

    # 2. CLIP 向量编码
    logger.info("[Stage 2.2] 开始 CLIP 编码...")
    clip = CLIPEncoder(model_name=config.model_clip)

    keyframe_paths = []
    for scene in scenes:
        if scene.keyframe_paths:
            keyframe_paths.append(scene.keyframe_paths[0])

    if keyframe_paths:
        embeddings = clip.encode_images(keyframe_paths, batch_size=config.clip_batch_size)

        for i, scene in enumerate(scenes):
            if i < len(embeddings):
                scene.clip_embedding = embeddings[i].tolist()

        # 保存 CLIP 向量
        embeddings_dir = os.path.join(output_dir, "embeddings", video_id)
        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(os.path.join(embeddings_dir, "clip_vectors.npy"), embeddings)
        logger.info(f"CLIP 编码完成: {len(embeddings)} 个场景向量")

    # 3. VLM 场景描述 (如果有 API Key)
    if config.dashscope_api_key:
        logger.info("[Stage 2.3] 开始 VLM 场景描述...")
        _generate_scene_captions(scenes, scene_transcripts, config)

    # 4. 保存 enriched metadata
    metadata = {
        "video_id": video_id,
        "scene_count": len(scenes),
        "scenes": [s.to_dict() for s in scenes],
        "transcripts": {sid: texts for sid, texts in scene_transcripts.items()},
    }
    scenes_dir = os.path.join(output_dir, "scenes", video_id)
    save_json(metadata, os.path.join(scenes_dir, "metadata.json"))
    logger.info("[Stage 2] 完成")


def _generate_scene_captions(
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    config: AppConfig,
):
    """使用 Qwen VL 生成场景描述"""
    from vl_core.llm.qwen_vl import QwenVLClient

    vl_client = QwenVLClient(model=config.model_vlm, api_key=config.dashscope_api_key)

    prompts = config.prompts.get("scene_caption", {})
    system_prompt = prompts.get("system", "你是一个专业的影视内容分析师。")
    user_template = prompts.get("user", "")

    prev_summary = ""
    total = len(scenes)

    for i, scene in enumerate(scenes):
        if not scene.keyframe_paths:
            continue

        image_path = scene.keyframe_paths[0]
        transcript = " ".join(scene_transcripts.get(scene.scene_id, []))

        if user_template:
            prompt = user_template.format(
                genre="影视",
                video_title=scene.video_id,
                scene_index=i + 1,
                total_scenes=total,
                prev_summary=prev_summary,
                transcript=transcript or "(无台词)",
            )
        else:
            prompt = f"请简要描述这个视频场景（第{i+1}/{total}个场景）。"
            if transcript:
                prompt += f"\n本场景台词：{transcript}"

        caption = vl_client.analyze_image(image_path, prompt)
        if caption:
            scene.vlm_caption = caption
            prev_summary = caption[:200]

        if (i + 1) % 5 == 0:
            logger.info(f"VLM 进度: {i + 1}/{total}")
