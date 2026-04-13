"""角色聚类 - 将人脸嵌入向量聚类为角色身份"""

import numpy as np

from vl.core.logging import get_logger
from vl.core.models.character import FaceDetection, Character

logger = get_logger()


class CharacterCluster:
    """基于余弦相似度的角色聚类器"""

    def cluster(
        self,
        detections: list[FaceDetection],
        threshold: float = 0.6,
    ) -> list[Character]:
        """将人脸检测结果聚类为角色"""
        if not detections:
            logger.warning("无人脸检测结果，跳过聚类")
            return []

        # 构建嵌入矩阵
        embeddings = np.array([d.embedding for d in detections])
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        # 贪心在线聚类
        cluster_centers = []  # 每个簇的归一化中心向量
        cluster_members = []  # 每个簇的检测索引列表

        for i, emb in enumerate(embeddings_norm):
            best_cluster = -1
            best_sim = 0.0

            for ci, center in enumerate(cluster_centers):
                sim = float(np.dot(emb, center))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = ci

            if best_sim >= threshold and best_cluster >= 0:
                cluster_members[best_cluster].append(i)
                # 更新簇中心（运行均值）
                n = len(cluster_members[best_cluster])
                cluster_centers[best_cluster] = (
                    cluster_centers[best_cluster] * (n - 1) + emb
                ) / n
                cluster_centers[best_cluster] /= np.linalg.norm(cluster_centers[best_cluster])
            else:
                cluster_centers.append(emb.copy())
                cluster_members.append([i])

        # 构建 Character 对象
        characters = []
        for ci, members in enumerate(cluster_members):
            # 选择最接近中心的人脸作为代表
            center = cluster_centers[ci]
            member_embs = embeddings_norm[members]
            sims = member_embs @ center
            best_local = int(np.argmax(sims))
            best_det = detections[members[best_local]]

            # 收集出场场景（去重）
            scene_ids = list(dict.fromkeys(
                detections[m].scene_id for m in members if detections[m].scene_id
            ))

            characters.append(Character(
                character_id=f"char_{ci:02d}",
                label=f"Character_{ci:02d}",
                representative_face_path=best_det.image_path,
                appearance_scenes=scene_ids,
            ))

        # 按出场次数排序
        characters.sort(key=lambda c: len(c.appearance_scenes), reverse=True)

        logger.info(f"角色聚类完成: {len(characters)} 个角色")
        for c in characters:
            logger.info(f"  {c.label}: {len(c.appearance_scenes)} 个场景")

        return characters
