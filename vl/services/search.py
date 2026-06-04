"""语义检索服务"""

import json
import os
import re

import numpy as np

from vl.core.config import AppConfig, get_config
from vl.core.helpers.json_utils import load_json
from vl.core.helpers.prompt_loader import load_prompt
from vl.vision.clip_encoder import CLIPEncoder
from vl.store.vector_store import VectorStore
from vl.core.logging import get_logger

logger = get_logger()


def search_scenes(
    query: str,
    video: str = "",
    top_k: int = 10,
) -> tuple[list[dict], AppConfig]:
    """CLIP 编码查询 → FAISS 检索相关场景。

    Returns:
        (results, config) — results 为匹配的场景列表
    """
    config = get_config()
    clip = CLIPEncoder(model_name=config.model_clip)

    # --- 查询扩展 ---
    queries = [query]
    if config.dashscope_api_key:
        try:
            from vl.core.llm.qwen_text import QwenTextClient
            qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

            user_tpl, sys_prompt = load_prompt(config, "query_expand")
            if user_tpl:
                expanded = qwen.generate(user_tpl.format(query=query), system=sys_prompt)
                if expanded:
                    extra = [w.strip() for w in expanded.split() if w.strip()]
                    queries.extend(extra)
                    logger.info(f"查询扩展: {query} -> {queries}")
        except Exception as e:
            logger.warning(f"查询扩展失败 (使用原始查询): {e}")

    # 编码查询
    query_vecs = clip.encode_texts(queries)
    query_vec = np.mean(query_vecs, axis=0)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # 查找索引
    index_root = os.path.join(config.output_root, "stage3_captions")
    if not os.path.isdir(index_root):
        return None, config

    all_results = []
    video_dirs = [video] if video else os.listdir(index_root)

    for vid in video_dirs:
        idx_dir = os.path.join(index_root, vid)
        faiss_path = os.path.join(idx_dir, "index.faiss")
        doc_path = os.path.join(idx_dir, "doc_store.json")

        if not os.path.isfile(faiss_path) or not os.path.isfile(doc_path):
            continue

        store = VectorStore(dim=clip.dim)
        store.load(faiss_path)

        results = store.search(query_vec, top_k=top_k)
        docs = load_json(doc_path)
        doc_map = {d["scene_id"]: d for d in docs}

        for scene_id, score in results:
            doc = doc_map.get(scene_id, {})
            all_results.append({
                "video_id": vid,
                "score": score,
                **doc,
            })

    # 排序
    all_results.sort(key=lambda x: -x["score"])
    all_results = all_results[:top_k]

    # LLM 重排
    if config.retrieval_rerank and config.dashscope_api_key and all_results:
        all_results = _rerank_results(query, all_results, config)

    return all_results, config


def _rerank_results(query: str, results: list[dict], config: AppConfig) -> list[dict]:
    """LLM 重排搜索结果"""
    try:
        from vl.core.llm.qwen_text import QwenTextClient
        qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

        user_tpl, sys_prompt = load_prompt(config, "rerank")

        if user_tpl:
            for r in results:
                prompt = user_tpl.format(
                    query=query,
                    scene_caption=r.get("vlm_caption", ""),
                    transcript=r.get("transcript", ""),
                )
                resp = qwen.generate(prompt, system=sys_prompt)
                nums = re.findall(r'\d+', resp or "")
                if nums:
                    r["rerank_score"] = int(nums[0])
                else:
                    r["rerank_score"] = r["score"] * 10

            results.sort(key=lambda x: -x.get("rerank_score", 0))
            logger.info("LLM 重排完成")
    except Exception as e:
        logger.warning(f"LLM 重排失败 (使用原始排序): {e}")

    return results
