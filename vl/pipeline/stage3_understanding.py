"""Stage 3: 视觉语义理解 - VLM 补充描述 + FAISS 索引

如果 Stage 2 (qwen-omni) 已经生成了视觉描述，则跳过 VLM 调用，
仅做 FAISS 索引构建和文档存储。

如果 Stage 2 没有视觉描述 (qwen/whisper 后端)，则仍使用 VLM 生成。
"""

import json
import os

import numpy as np

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.helpers.json_utils import save_json, load_json
from vl.store.vector_store import VectorStore

from vl.core.logging import get_logger

logger = get_logger()


def run_stage3(
    video_id: str,
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    output_dir: str,
    config: AppConfig,
):
    """
    执行 Stage 3: 视觉语义理解 + 索引构建。

    1. VLM 结构化场景描述 (仅在 Stage 2 未生成时)
    2. FAISS 向量索引构建
    3. 文档存储 (enriched with structured captions)
    """
    # --- 3.1 检查是否需要 VLM 补充描述 ---
    scenes_with_caption = sum(1 for s in scenes if s.structured_caption)
    needs_vlm = scenes_with_caption < len(scenes) // 2  # 超过一半没描述才补充

    if needs_vlm and config.dashscope_api_key:
        logger.info("[Stage 3.1] 场景描述不完整 (%d/%d)，使用 VLM 补充...",
                     scenes_with_caption, len(scenes))
        _generate_structured_captions(
            scenes=scenes,
            scene_transcripts=scene_transcripts,
            config=config,
        )
    else:
        logger.info("[Stage 3.1] 已有 %d 个场景描述，跳过 VLM", scenes_with_caption)

    # --- 3.2 保存结构化描述 ---
    captions_data = []
    for scene in scenes:
        entry = {
            "scene_id": scene.scene_id,
            "index": scene.index,
            "start_time": scene.start_time,
            "end_time": scene.end_time,
        }
        if scene.structured_caption:
            entry["structured_caption"] = scene.structured_caption
        if scene.vlm_caption:
            entry["vlm_caption"] = scene.vlm_caption
        captions_data.append(entry)

    captions_dir = os.path.join(output_dir, "stage3_captions", video_id)
    os.makedirs(captions_dir, exist_ok=True)
    save_json(captions_data, os.path.join(captions_dir, "captions.json"))
    logger.info(f"结构化描述已保存: {len(captions_data)} 个场景")

    # --- 3.3 构建 FAISS 索引 ---
    embeddings_path = os.path.join(output_dir, "stage2_features", video_id, "clip_vectors.npy")
    if os.path.isfile(embeddings_path):
        _build_faiss_index(
            video_id=video_id,
            scenes=scenes,
            scene_transcripts=scene_transcripts,
            embeddings_path=embeddings_path,
            output_dir=output_dir,
        )
    else:
        logger.warning("未找到 CLIP 向量文件，跳过索引构建")

    # --- 3.4 保存 enriched metadata ---
    metadata = {
        "video_id": video_id,
        "scene_count": len(scenes),
        "scenes": [s.to_dict() for s in scenes],
        "transcripts": {sid: texts for sid, texts in scene_transcripts.items()},
    }
    scenes_dir = os.path.join(output_dir, "stage1_scenes", video_id)
    save_json(metadata, os.path.join(scenes_dir, "metadata.json"))

    logger.info("[Stage 3] 完成")


def _generate_structured_captions(
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    config: AppConfig,
):
    """使用 VLM 生成结构化场景描述 (仅在需要时调用)"""
    from vl.core.llm.qwen_vl import QwenVLClient
    from vl.core.llm.base_client import BaseLLMClient

    vl_client = QwenVLClient(model=config.model_vlm, api_key=config.dashscope_api_key)
    json_extractor = BaseLLMClient(model=config.model_text, api_key=config.dashscope_api_key)

    prompts = config.prompts.get("scene_caption", {})
    user_template = prompts.get("user", "")

    prev_caption = "{}"
    total = len(scenes)
    captioned = 0

    for i, scene in enumerate(scenes):
        # 跳过已有描述的场景
        if scene.structured_caption:
            continue
        if not scene.keyframe_paths:
            continue

        logger.info("VLM 补充描述: %d/%d", i + 1, total)

        audiotext = " ".join(scene_transcripts.get(scene.scene_id, [])) or "(无台词)"

        if not user_template:
            logger.warning("scene_caption prompt 未配置，跳过 VLM 补充")
            return

        prompt = user_template.format(
            audiotext=audiotext,
            caption=prev_caption,
        )

        raw_output = vl_client.analyze_images(
            scene.keyframe_paths, prompt,
            window_size=config.vlm_window_size,
            stride=config.vlm_stride,
        )
        if raw_output:
            extracted = json_extractor.extract_json(raw_output)
            if extracted:
                try:
                    caption_dict = json.loads(extracted)
                    # 归一化: actions(list) → main_actions(string)
                    if "actions" in caption_dict and "main_actions" not in caption_dict:
                        actions = caption_dict["actions"]
                        if isinstance(actions, list):
                            caption_dict["main_actions"] = "；".join(str(a) for a in actions)
                    # 归一化: interaction → interactions
                    if "interaction" in caption_dict and "interactions" not in caption_dict:
                        caption_dict["interactions"] = caption_dict["interaction"]
                    scene.structured_caption = caption_dict
                    prev_caption = extracted
                    captioned += 1
                except json.JSONDecodeError:
                    scene.vlm_caption = raw_output
                    prev_caption = raw_output[:300]
                    captioned += 1
            else:
                scene.vlm_caption = raw_output
                prev_caption = raw_output[:300]
                captioned += 1

    logger.info(f"VLM 补充描述完成: 共生成 {captioned} 个")


def _build_faiss_index(
    video_id: str,
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    embeddings_path: str,
    output_dir: str,
):
    """构建 FAISS 向量索引 + 文档存储"""
    logger.info("[Stage 3.3] 开始构建 FAISS 索引...")

    embeddings = np.load(embeddings_path)
    logger.info(f"加载 CLIP 向量: shape={embeddings.shape}")

    indexed_scenes = [s for s in scenes if s.clip_embedding is not None]
    scene_ids = [s.scene_id for s in indexed_scenes]

    if not scene_ids:
        logger.warning("没有可索引的场景")
        return

    dim = embeddings.shape[1]
    store = VectorStore(dim=dim)
    store.add_batch(scene_ids, embeddings)
    logger.info(f"索引向量总数: {store.size}")

    index_dir = os.path.join(output_dir, "stage3_captions", video_id)
    os.makedirs(index_dir, exist_ok=True)
    store.save(os.path.join(index_dir, "index.faiss"))

    doc_store = []
    for scene in indexed_scenes:
        doc = {
            "scene_id": scene.scene_id,
            "video_id": scene.video_id,
            "index": scene.index,
            "start_time": scene.start_time,
            "end_time": scene.end_time,
            "keyframe_paths": scene.keyframe_paths,
            "vlm_caption": scene.vlm_caption,
            "transcript": " ".join(scene_transcripts.get(scene.scene_id, [])),
        }
        if scene.structured_caption:
            doc["structured_caption"] = scene.structured_caption
        doc_store.append(doc)

    save_json(doc_store, os.path.join(index_dir, "doc_store.json"))

    logger.info(f"索引构建完成: {store.size} 个场景向量, dim={dim}")
