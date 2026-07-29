"""混合检索: 向量 (text-embedding-v3) + BM25 + RRF 融合 + 时间邻域.

embedding 持久化到 {output}/{video_dir}/retrieval_emb.npz (首次建, 后续加载).
向量检索用 numpy 内存余弦 (362 文档量级, <1ms, 不需要向量库).
RRF 融合 BM25 + 向量双路 rank, 兼顾关键词精确匹配 + 语义匹配.
"""

from __future__ import annotations

import functools
import os
import numpy as np

from src.core.config import get_config
from src.core.helpers.json_utils import load_json


# ============================================================
# Embedding (DashScope qwen3.7-text-embedding)
# ============================================================

def _embed_texts(texts: list[str]) -> np.ndarray:
    """调 DashScope 批量 embedding. 返回 (N, D) float32, D 由模型决定.

    模型名来自 config/pipeline.yaml: models.embedding (cfg.model_embedding).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    import dashscope

    cfg = get_config()
    dashscope.api_key = cfg.dashscope_api_key
    embed_model = cfg.model_embedding

    all_emb: list[list[float]] = []
    batch_size = 10  # DashScope embedding batch 上限
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = dashscope.TextEmbedding.call(model=embed_model, input=batch)
        if resp.status_code != 200:
            raise RuntimeError(
                f"{embed_model} 调用失败: status={resp.status_code}, "
                f"code={resp.code}, msg={resp.message}"
            )
        for item in resp.output["embeddings"]:
            all_emb.append(item["embedding"])
    return np.array(all_emb, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """单条 query → (1024,) 向量."""
    return _embed_texts([query])[0]


# ============================================================
# 索引构建与持久化
# ============================================================

def _cache_path(video_dir: str) -> str:
    cfg = get_config()
    return os.path.join(cfg.output_root, video_dir, "retrieval_emb.npz")


@functools.lru_cache(maxsize=8)
def build_or_load_embeddings(
    video_dir: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """构建或加载一集的 embedding (events.retrieval_text + segments.text).

    首次调用时 embedding (慢, ~3-10s), 持久化 .npz; 后续加载 (<100ms).
    events/segments 数量变化 (重新建库) 时自动重建.
    内存缓存: 同 video_dir 的多次调用直接返回 (lru_cache, maxsize=8).
    调用方必须把返回的 ndarray 当只读 (vector_search 等只做矩阵乘法, 安全).

    Returns:
        (events_emb, segs_emb, events_text, segs_text)
        events_emb: (N_events, 1024); segs_emb: (N_segs, 1024)
    """
    cache = _cache_path(video_dir)
    cfg = get_config()
    ep_dir = os.path.join(cfg.output_root, video_dir)
    from src.eval.stage3_retrieval import build_searchable_text

    events = load_json(os.path.join(ep_dir, "stage3_dryrun.json")).get("events", []) or []
    segments = load_json(os.path.join(ep_dir, "audio.json")).get("segments", []) or []
    events_text = [build_searchable_text(e) for e in events]
    segs_text = [s.get("text", "") for s in segments]

    # 快速路径: 缓存存在 + 数量匹配 → 直接加载
    if os.path.isfile(cache):
        data = np.load(cache)
        if (
            int(data["events_emb"].shape[0]) == len(events_text)
            and int(data["segs_emb"].shape[0]) == len(segs_text)
        ):
            return data["events_emb"], data["segs_emb"], events_text, segs_text

    # 慢路径: 首次建 embedding
    print(
        f"[retriever] 建 embedding: {len(events_text)} events + "
        f"{len(segs_text)} segments (首次, 后续走缓存)",
        flush=True,
    )
    events_emb = _embed_texts(events_text)
    segs_emb = _embed_texts(segs_text)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, events_emb=events_emb, segs_emb=segs_emb)
    print(f"[retriever] embedding 已缓存: {cache}", flush=True)
    return events_emb, segs_emb, events_text, segs_text


# ============================================================
# 向量检索 + RRF 融合
# ============================================================

def vector_search(
    query_emb: np.ndarray, doc_emb: np.ndarray, top_k: int = 5,
) -> list[tuple[int, float]]:
    """numpy 余弦相似度检索.

    Returns: [(doc_idx, cosine_score), ...] score ∈ [0,1], 降序.
    """
    if doc_emb.shape[0] == 0:
        return []
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    d = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)
    scores = d @ q  # (N,) 余弦相似度
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(int(i), float(scores[i])) for i in top_idx]


def rrf_fuse(
    *ranked_lists: list[tuple[int, float]], k: int = 60, top_k: int = 5,
) -> list[tuple[int, float]]:
    """RRF (Reciprocal Rank Fusion) 融合多路 rank.

    每路 ranked_list = [(idx, score), ...] (按 score 降序).
    融合公式: fused(idx) = Σ 1/(k + rank_in_list + 1).
    默认 k=60 (业界经验值).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
