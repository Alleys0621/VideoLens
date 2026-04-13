"""Stage 2: 多维分析 - ASR + CLIP + VLM 场景描述"""

import os
import numpy as np

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.helpers.json_utils import save_json
from vl.asr.transcriber import Transcriber
from vl.vision.clip_encoder import CLIPEncoder

from vl.core.logging import get_logger

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
    logger.info("转录中... 可能需要几分钟")
    transcriber = Transcriber(
        model_size=config.model_whisper,
        language=config.asr_language,
        beam_size=config.asr_beam_size,
        vad_filter=config.asr_vad_filter,
    )
    segments = transcriber.transcribe(audio_path)
    logger.info(f"ASR 返回 {len(segments)} 个转录片段")
    segments = transcriber.assign_to_scenes(segments, scenes)
    logger.info(f"场景分配完成: {len(segments)} 个片段已分配到对应场景")

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

    # --- Phase 2: 说话人识别 + 对齐 ---
    try:
        from vl.asr.diarizer import Diarizer
        from vl.asr.aligner import Aligner

        logger.info("[Stage 2.1b] 开始说话人识别...")
        diarizer = Diarizer(model_name=config.model_diarization)
        diarization = diarizer.diarize(audio_path)

        aligner = Aligner()
        segments = aligner.align(segments, diarization)

        # 更新保存的转录结果
        save_json(
            [s.to_dict() for s in segments],
            os.path.join(transcript_dir, "transcript.json"),
        )
        # 更新场景台词映射
        scene_transcripts.clear()
        for seg in segments:
            scene_transcripts.setdefault(seg.scene_id, []).append(seg.text)
    except Exception as e:
        logger.warning(f"说话人识别失败 (跳过): {e}")

    # 2. CLIP 向量编码
    logger.info("[Stage 2.2] 开始 CLIP 编码...")
    clip = CLIPEncoder(model_name=config.model_clip)

    keyframe_paths = []
    for scene in scenes:
        if scene.keyframe_paths:
            keyframe_paths.append(scene.keyframe_paths[0])

    logger.info(f"准备编码 {len(keyframe_paths)} 张关键帧图片")
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

    # --- Phase 2: 人脸检测 + 角色聚类 ---
    characters = []
    try:
        from vl.vision.face_detector import FaceDetector
        from vl.vision.face_cluster import CharacterCluster

        logger.info("[Stage 2.2b] 开始人脸检测...")
        face_detector = FaceDetector(detection_threshold=config.face_detection_threshold)

        kf_paths = [s.keyframe_paths[0] for s in scenes if s.keyframe_paths]
        kf_scene_ids = [s.scene_id for s in scenes if s.keyframe_paths]

        all_detections = face_detector.detect_batch(kf_paths, kf_scene_ids)

        if all_detections:
            logger.info("[Stage 2.2c] 开始角色聚类...")
            clusterer = CharacterCluster()
            characters = clusterer.cluster(all_detections, threshold=config.face_clustering_threshold)

            characters_dir = os.path.join(output_dir, "characters", video_id)
            os.makedirs(characters_dir, exist_ok=True)
            save_json(
                [c.to_dict() for c in characters],
                os.path.join(characters_dir, "characters.json"),
            )
    except Exception as e:
        logger.warning(f"人脸检测/聚类失败 (跳过): {e}")

    # 3. VLM 场景描述 (如果有 API Key)
    if config.dashscope_api_key:
        logger.info("[Stage 2.3] 开始 VLM 场景描述... 共 %d 个场景待处理", len(scenes))
        _generate_scene_captions(scenes, scene_transcripts, config)
    else:
        logger.info("[Stage 2.3] 未配置 DashScope API Key，跳过 VLM 场景描述")

    # 4. 保存 enriched metadata
    metadata = {
        "video_id": video_id,
        "scene_count": len(scenes),
        "scenes": [s.to_dict() for s in scenes],
        "transcripts": {sid: texts for sid, texts in scene_transcripts.items()},
        "characters": [c.to_dict() for c in characters] if characters else [],
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
    from vl.core.llm.qwen_vl import QwenVLClient

    vl_client = QwenVLClient(model=config.model_vlm, api_key=config.dashscope_api_key)

    prompts = config.prompts.get("scene_caption", {})
    system_prompt = prompts.get("system", "你是一个专业的影视内容分析师。")
    user_template = prompts.get("user", "")

    prev_summary = ""
    total = len(scenes)
    captioned = 0

    for i, scene in enumerate(scenes):
        if not scene.keyframe_paths:
            continue

        logger.info("生成场景描述: %d/%d", i + 1, total)

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
            captioned += 1

        if (i + 1) % 5 == 0:
            logger.info(f"VLM 进度: {i + 1}/{total} (已生成 {captioned} 个描述)")

    logger.info(f"VLM 场景描述完成: 共生成 {captioned} 个描述，总计 {total} 个场景")
