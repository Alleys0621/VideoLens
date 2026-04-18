"""Stage 2: 多模态场景理解

使用 Qwen3.5-Omni-Plus 同时理解音频+画面，像人一样看视频：
  - 听到谁在说话、说了什么、什么情感
  - 看到画面中有什么角色、在做什么、什么场景
  - 音频与画面的自然关联

流程:
  1. 将场景按时间分组 (~2min/组)
  2. 每组: 音频切片 + 关键帧 → Qwen3.5-Omni → 结构化理解
  3. 解析说话人/台词/情感/视觉描述 → 映射回场景
  4. CLIP 向量编码 (用于检索)
"""

import os
import numpy as np

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.models.transcript import TranscriptSegment
from vl.core.helpers.json_utils import save_json
from vl.vision.clip_encoder import CLIPEncoder
from pydub import AudioSegment

from vl.core.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# 场景分组: 按时间将场景合并为 ~target_duration 秒的片段
# ---------------------------------------------------------------------------

def _group_scenes_by_time(
    scenes: list[Scene],
    target_duration: float = 120.0,
) -> list[list[Scene]]:
    """将场景按时间顺序分组，每组约 target_duration 秒。"""
    if not scenes:
        return []

    groups = []
    current_group = [scenes[0]]
    current_start = scenes[0].start_time

    for scene in scenes[1:]:
        elapsed = scene.end_time - current_start
        if elapsed > target_duration and current_group:
            groups.append(current_group)
            current_group = [scene]
            current_start = scene.start_time
        else:
            current_group.append(scene)

    if current_group:
        groups.append(current_group)

    return groups


def _select_keyframes(
    scenes: list[Scene],
    max_frames: int = 5,
) -> list[str]:
    """从一组场景中选择代表性关键帧 (均匀采样)。"""
    all_kf = []
    for s in scenes:
        if s.keyframe_paths:
            all_kf.append(s.keyframe_paths[0])

    if len(all_kf) <= max_frames:
        return all_kf

    # 均匀采样
    step = len(all_kf) / max_frames
    return [all_kf[int(i * step)] for i in range(max_frames)]


# ---------------------------------------------------------------------------
# Qwen-Omni 后端: 全模态理解
# ---------------------------------------------------------------------------

def _transcribe_qwen_omni(
    audio_path: str,
    scenes: list[Scene],
    config: AppConfig,
) -> tuple[list[TranscriptSegment], dict[str, dict], dict[str, str]]:
    """
    使用 Qwen3.5-Omni-Plus 进行全模态理解。

    Returns:
        segments: 转录片段列表
        visual_descriptions: {scene_id: visual_dict} 视觉描述
        scene_content_types: {scene_id: content_type} 内容类型
    """
    from vl.asr.qwen_omni import QwenOmni

    omni = QwenOmni(
        api_key=config.dashscope_api_key,
        model=config.model_omni,
        language=config.asr_language,
    )

    # 按时间分组
    groups = _group_scenes_by_time(scenes, target_duration=config.asr_chunk_duration)
    logger.info(f"场景分为 {len(groups)} 个片段 (每组 ~{config.asr_chunk_duration}s)")

    # 加载完整音频
    logger.info(f"加载音频: {audio_path}")
    full_audio = AudioSegment.from_file(audio_path)
    logger.info(f"音频时长: {len(full_audio) / 1000:.1f}秒")

    all_segments: list[TranscriptSegment] = []
    visual_descriptions: dict[str, dict] = {}
    scene_content_types: dict[str, str] = {}

    for i, group in enumerate(groups):
        group_start = group[0].start_time
        group_end = group[-1].end_time
        logger.info(f"处理片段 {i + 1}/{len(groups)}: "
                     f"场景 {group[0].scene_id}~{group[-1].scene_id} "
                     f"[{group_start:.1f}s - {group_end:.1f}s]")

        # 提取音频切片
        start_ms = int(group_start * 1000)
        end_ms = int(group_end * 1000)
        chunk_audio = full_audio[start_ms:end_ms]

        tmp_path = os.path.join(
            os.path.dirname(audio_path),
            f"_omni_chunk_{i}.wav",
        )
        chunk_audio.export(tmp_path, format="wav")

        # 选择关键帧
        kf_paths = _select_keyframes(group, config.asr_max_keyframes_per_chunk)

        try:
            result = omni.understand_chunk(
                audio_path=tmp_path,
                image_paths=kf_paths,
                time_offset=group_start,
                chunk_index=i,
                time_end=group_end,
            )

            # 将台词转为 TranscriptSegment
            for seg in result.segments:
                # 找到对应的场景
                mid = (seg.start_time + seg.end_time) / 2
                scene_id = ""
                for s in group:
                    if s.start_time <= mid <= s.end_time:
                        scene_id = s.scene_id
                        break

                all_segments.append(TranscriptSegment(
                    segment_id=seg.segment_id,
                    scene_id=scene_id,
                    speaker_id=seg.speaker or "SPEAKER_00",
                    text=seg.text,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    confidence=seg.confidence,
                ))

            # 视觉描述映射到组内场景
            if result.scene_description:
                vis_dict = result.scene_description.to_dict()
                # 主要映射到组内中间场景
                mid_idx = len(group) // 2
                for j, s in enumerate(group):
                    # 所有场景共享基础描述，但可以后续被 stage4 细化
                    visual_descriptions[s.scene_id] = vis_dict

            # 将 content_type 映射到组内场景
            for s in group:
                scene_content_types[s.scene_id] = result.content_type

            # 保存说话人信息
            if result.speakers:
                speakers_path = os.path.join(
                    os.path.dirname(audio_path),
                    f"_omni_speakers_{i}.json",
                )
                save_json(result.speakers, speakers_path)

        except Exception as e:
            logger.warning(f"片段 {i + 1} 理解失败: {e}")
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    logger.info(f"全模态理解完成: {len(all_segments)} 句台词, "
                 f"{len(visual_descriptions)} 个场景有视觉描述")
    # 统计 content_type 分布
    ct_counts = {}
    for ct in scene_content_types.values():
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    logger.info(f"内容类型分布: {ct_counts}")
    return all_segments, visual_descriptions, scene_content_types


# ---------------------------------------------------------------------------
# Qwen-ASR 后端: 纯音频转录 (旧版, 保留兼容)
# ---------------------------------------------------------------------------

def _transcribe_qwen(
    audio_path: str,
    scenes: list[Scene],
    config: AppConfig,
) -> tuple[list[TranscriptSegment], dict]:
    from vl.asr.qwen_asr import QwenASR

    asr = QwenASR(
        api_key=config.dashscope_api_key,
        language=config.asr_language,
        chunk_duration=config.asr_chunk_duration,
        silence_min_len=config.asr_silence_min_len,
        silence_thresh=config.asr_silence_thresh,
    )

    qwen_segments = asr.transcribe(audio_path)
    qwen_segments = asr.assign_to_scenes(qwen_segments, scenes)

    results = []
    for seg in qwen_segments:
        results.append(TranscriptSegment(
            segment_id=seg.segment_id,
            scene_id=seg.scene_id,
            speaker_id=seg.speaker or "SPEAKER_00",
            text=seg.text,
            start_time=seg.start_time,
            end_time=seg.end_time,
            confidence=seg.confidence,
        ))
    return results, {}


# ---------------------------------------------------------------------------
# Whisper 本地后端 (旧版, 保留兼容)
# ---------------------------------------------------------------------------

def _transcribe_whisper(
    audio_path: str,
    scenes: list[Scene],
    config: AppConfig,
) -> tuple[list[TranscriptSegment], dict]:
    from vl.asr.transcriber import Transcriber

    transcriber = Transcriber(
        model_size=config.model_whisper,
        language=config.asr_language,
        beam_size=config.asr_beam_size,
        vad_filter=config.asr_vad_filter,
    )
    segments = transcriber.transcribe(audio_path)
    segments = transcriber.assign_to_scenes(segments, scenes)
    return segments, {}


# ---------------------------------------------------------------------------
# Stage 2 主入口
# ---------------------------------------------------------------------------

def run_stage2(
    video_path: str,
    video_id: str,
    scenes: list[Scene],
    audio_path: str,
    output_dir: str,
    config: AppConfig,
) -> dict[str, list[str]]:
    """
    执行 Stage 2: 多模态场景理解。

    根据配置选择后端:
      - qwen-omni: 全模态 (音频+画面) 理解，推荐
      - qwen: 纯音频 API 转录
      - whisper: 本地 faster-whisper

    Returns:
        scene_transcripts: {scene_id: [text_segments]}
    """
    backend = config.asr_backend
    logger.info("[Stage 2.1] 开始多模态理解... (backend=%s)", backend)

    if backend == "qwen-omni" and config.dashscope_api_key:
        segments, visual_descs, scene_content_types = _transcribe_qwen_omni(audio_path, scenes, config)
    elif backend == "qwen" and config.dashscope_api_key:
        segments, visual_descs = _transcribe_qwen(audio_path, scenes, config)
        scene_content_types = {s.scene_id: "main" for s in scenes}
    else:
        segments, visual_descs = _transcribe_whisper(audio_path, scenes, config)
        scene_content_types = {s.scene_id: "main" for s in scenes}

    # 保存转录结果
    transcript_dir = os.path.join(output_dir, "transcripts", video_id)
    os.makedirs(transcript_dir, exist_ok=True)
    save_json(
        [s.to_dict() for s in segments],
        os.path.join(transcript_dir, "transcript.json"),
    )
    logger.info(f"转录完成: {len(segments)} 个片段")

    # 将视觉描述写入场景
    if visual_descs:
        for scene in scenes:
            if scene.scene_id in visual_descs:
                vis = visual_descs[scene.scene_id]
                # 合并到 structured_caption (如果还没有的话)
                if not scene.structured_caption:
                    scene.structured_caption = {}
                scene.structured_caption.update(vis)

    # 将 content_type 写入场景
    for scene in scenes:
        if scene.scene_id in scene_content_types:
            scene.content_type = scene_content_types[scene.scene_id]

    # 构建场景到台词的映射
    scene_transcripts: dict[str, list[str]] = {}
    for seg in segments:
        scene_transcripts.setdefault(seg.scene_id, []).append(seg.text)

    # --- 说话人对齐 (仅 whisper 后端需要) ---
    # 已移除: 使用 qwen-omni 全模态后端时，说话人识别由模型直接完成

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

        embeddings_dir = os.path.join(output_dir, "embeddings", video_id)
        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(os.path.join(embeddings_dir, "clip_vectors.npy"), embeddings)
        logger.info(f"CLIP 编码完成: {len(embeddings)} 个场景向量")

    # 3. 从 omni 输出中提取角色信息
    _extract_characters_from_omni(
        visual_descriptions=visual_descs,
        scene_transcripts=scene_transcripts,
        scenes=scenes,
        video_id=video_id,
        output_dir=output_dir,
        config=config,
    )

    logger.info("[Stage 2] 完成")
    return scene_transcripts


def _extract_characters_from_omni(
    visual_descriptions: dict[str, dict],
    scene_transcripts: dict[str, list[str]],
    scenes: list[Scene],
    video_id: str,
    output_dir: str,
    config: AppConfig,
):
    """从 qwen-omni 输出中提取角色信息，保存到 characters.json"""
    from collections import defaultdict
    from vl.core.models.character import Character

    logger.info("[Stage 2.3] 从 omni 输出中提取角色信息...")

    # 1. 从 visual_descriptions 中聚合角色名 → 出现场景
    char_scenes: dict[str, set[str]] = defaultdict(set)
    for scene in scenes:
        vis = visual_descriptions.get(scene.scene_id)
        if not vis:
            continue
        for char_name in vis.get("characters", []):
            if char_name and char_name not in ("未知", "无"):
                char_scenes[char_name].add(scene.scene_id)

    # 2. 可选: 从台词中提取额外角色名 (LLM)
    if config.dashscope_api_key and scene_transcripts:
        try:
            from vl.core.llm.qwen_text import QwenTextClient

            all_transcript = []
            for texts in scene_transcripts.values():
                all_transcript.extend(texts)
            transcript_summary = " ".join(all_transcript[:500])

            if transcript_summary.strip():
                text_client = QwenTextClient(
                    model=config.model_text, api_key=config.dashscope_api_key
                )
                existing_names = list(char_scenes.keys())
                prompt = (
                    f"以下是一部视频的台词片段：\n"
                    f"{transcript_summary}\n\n"
                    f"已识别的角色：{', '.join(existing_names) if existing_names else '无'}\n\n"
                    f"请列出台词中出现的所有角色名，每行一个，只输出名字。"
                )
                raw = text_client.generate(prompt)
                if raw:
                    extra_names = [
                        n.strip()
                        for n in raw.strip().split("\n")
                        if n.strip() and n.strip() not in char_scenes
                    ]
                    for name in extra_names:
                        char_scenes[name]  # 创建空集合条目
                    if extra_names:
                        logger.info(f"  台词中额外识别的角色: {extra_names}")
        except Exception as e:
            logger.warning(f"  台词角色提取失败: {e}")

    # 3. 构建 Character 列表
    characters = []
    for idx, (name, scene_ids) in enumerate(sorted(char_scenes.items())):
        characters.append(Character(
            character_id=f"char_{idx:02d}",
            label=name,
            appearance_scenes=sorted(scene_ids),
        ))

    # 4. 保存
    if characters:
        characters_dir = os.path.join(output_dir, "characters", video_id)
        os.makedirs(characters_dir, exist_ok=True)
        save_json(
            [c.to_dict() for c in characters],
            os.path.join(characters_dir, "characters.json"),
        )
        logger.info(f"角色提取完成: {[c.label for c in characters]}")
    else:
        logger.info("未识别到角色")
