"""FAISS 向量索引"""

import os

import faiss
import numpy as np


class VectorStore:
    """基于 FAISS 的向量存储"""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 内积 (配合归一化向量 = 余弦相似度)
        self.id_map: list[str] = []          # scene_id 列表

    def add(self, scene_id: str, embedding: np.ndarray):
        """添加单个向量"""
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.id_map.append(scene_id)

    def add_batch(self, scene_ids: list[str], embeddings: np.ndarray):
        """批量添加向量"""
        vecs = embeddings.astype(np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self.index.add(vecs)
        self.id_map.extend(scene_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """搜索最相似的向量，返回 (scene_id, score) 列表"""
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(vec, min(top_k, len(self.id_map)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        return results

    def save(self, path: str):
        """保存索引到文件（兼容中文路径）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # faiss 不支持中文路径，先写到临时文件再重命名
        import tempfile, shutil
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
        os.close(tmp_fd)
        try:
            faiss.write_index(self.index, tmp_path)
            shutil.move(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

        # 保存 id_map
        import json
        map_path = path.replace(".faiss", "_id_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False)

    def load(self, path: str):
        """从文件加载索引（兼容中文路径）"""
        import tempfile, shutil
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
        os.close(tmp_fd)
        try:
            shutil.copy2(path, tmp_path)
            self.index = faiss.read_index(tmp_path)
        finally:
            os.unlink(tmp_path)

        import json
        map_path = path.replace(".faiss", "_id_map.json")
        with open(map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

    @property
    def size(self) -> int:
        return self.index.ntotal
