"""Stage 3: 视觉语义理解 - VLM 补充描述 + FAISS 索引"""

import json
import numpy as np

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.paths import PathManager
from vl.core.helpers.json_utils import save_json
from vl.core.helpers.text_utils import extract_json
from vl.core.helpers.prompt_loader import load_prompt
from vl.store.vector_store import VectorStore

from vl.core.logging import get_logger

logger = get_logger()


def run_stage3(
    video_id: str,
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    paths: PathManager,
    config: AppConfig,
):
    """执行 Stage 3: 视觉语义理解 + 索引构建。"""

    # --- 3.1 检查是否需要 VLM 补充描述 ---
    scenes_with_caption = sum(1 for s in scenes if s.structured_caption)
    needs_vlm = scenes_with_caption < len(scenes) // 2

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

    save_json(captions_data, paths.captions_json_path)
    logger.info(f"结构化描述已保存: {len(captions_data)} 个场景")

    # --- 3.3 构建 FAISS 索引 ---
    if paths.clip_vectors_path_exists:
        _build_faiss_index(
            video_id=video_id,
            scenes=scenes,
            scene_transcripts=scene_transcripts,
            embeddings_path=paths.clip_vectors_path,
            paths=paths,
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
    save_json(metadata, paths.metadata_json_path)

    logger.info("[Stage 3] 完成")


def _generate_structured_captions(
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    config: AppConfig,
):
    """使用 VLM 生成结构化场景描述"""
    from vl.core.llm.qwen_vl import QwenVLClient
    from vl.core.llm.qwen_text import QwenTextClient

    vl_client = QwenVLClient(model=config.model_vlm, api_key=config.dashscope_api_key)
    text_client = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

    user_template, _ = load_prompt(config, "scene_caption")

    prev_caption = "{}"
    total = len(scenes)
    captioned = 0

    for i, scene in enumerate(scenes):
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
            extracted = text_client.extract_json(raw_output)
            if extracted:
                try:
                    caption_dict = json.loads(extracted)
                    scene.structured_caption = scene.get_normalized_caption()
                    # 用原始提取结果更新到场景
                    if "actions" in caption_dict and "main_actions" not in caption_dict:
                        actions = caption_dict["actions"]
                        if isinstance(actions, list):
                            caption_dict["main_actions"] = "\uff1b".join(str(a) for a in actions)
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
    paths: PathManager,
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

    store.save(paths.faiss_index_path)

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

    save_json(doc_store, paths.doc_store_path)

    logger.info(f"索引构建完成: {store.size} 个场景向量, dim={dim}")
