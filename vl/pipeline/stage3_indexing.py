"""Stage 3: 索引构建 - FAISS 向量索引 + 元数据"""

import os
import numpy as np

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.helpers.json_utils import load_json
from vl.store.vector_store import VectorStore

from vl.core.logging import get_logger

logger = get_logger()


def run_stage3(
    video_id: str,
    scenes: list[Scene],
    output_dir: str,
    config: AppConfig,
):
    """
    执行 Stage 3: 构建向量索引和文档存储。

    - 加载 CLIP 向量
    - 构建 FAISS 索引
    - 保存文档存储
    """
    logger.info("[Stage 3] 开始构建索引...")

    # 1. 加载 CLIP 向量
    embeddings_path = os.path.join(output_dir, "embeddings", video_id, "clip_vectors.npy")
    if not os.path.isfile(embeddings_path):
        logger.warning("未找到 CLIP 向量文件，跳过索引构建")
        return

    embeddings = np.load(embeddings_path)
    logger.info(f"加载 CLIP 向量: shape={embeddings.shape}")

    # 只索引有 embedding 的场景
    indexed_scenes = [s for s in scenes if s.clip_embedding is not None]
    scene_ids = [s.scene_id for s in indexed_scenes]

    if not scene_ids:
        logger.warning("没有可索引的场景")
        return

    # 2. 构建 FAISS 索引
    dim = embeddings.shape[1]
    logger.info(f"构建 FAISS 索引: dim={dim}, 向量数={len(scene_ids)}")
    store = VectorStore(dim=dim)
    store.add_batch(scene_ids, embeddings)
    logger.info(f"索引向量总数: {store.size}")

    # 3. 保存索引
    index_dir = os.path.join(output_dir, "index", video_id)
    os.makedirs(index_dir, exist_ok=True)
    faiss_path = os.path.join(index_dir, "index.faiss")
    store.save(faiss_path)

    # 4. 保存文档存储（用于检索结果的元数据查询）
    # 加载转录文本供重排使用
    transcript_map = {}
    transcript_path = os.path.join(output_dir, "transcripts", video_id, "transcript.json")
    if os.path.isfile(transcript_path):
        for seg in load_json(transcript_path):
            transcript_map.setdefault(seg.get("scene_id", ""), []).append(seg.get("text", ""))

    doc_store = []
    for scene in indexed_scenes:
        doc_store.append({
            "scene_id": scene.scene_id,
            "video_id": scene.video_id,
            "index": scene.index,
            "start_time": scene.start_time,
            "end_time": scene.end_time,
            "keyframe_paths": scene.keyframe_paths,
            "vlm_caption": scene.vlm_caption,
            "transcript": " ".join(transcript_map.get(scene.scene_id, [])),
        })

    from vl.core.helpers.json_utils import save_json
    doc_store_path = os.path.join(index_dir, "doc_store.json")
    save_json(doc_store, doc_store_path)

    logger.info(f"索引构建完成: {store.size} 个场景向量, dim={dim}")
    logger.info(f"FAISS 索引已保存: {faiss_path}")
    logger.info(f"文档存储已保存: {doc_store_path}")
