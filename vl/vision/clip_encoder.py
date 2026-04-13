"""CLIP 语义编码器 - 生成场景和文本的语义向量"""

import numpy as np
import os
from sentence_transformers import SentenceTransformer

from vl.core.logging import get_logger

logger = get_logger()


class CLIPEncoder:
    """CLIP 图文语义编码器"""

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        from vl.core.config import get_config
        # 触发 .env 加载，确保 HF_ENDPOINT 等环境变量已设置
        get_config()
        logger.info(f"加载 CLIP 模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        if self.dim is None:
            # CLIP 模型可能返回 None，用实际编码获取维度
            dummy = self.model.encode("test", normalize_embeddings=True)
            self.dim = len(dummy)
        logger.info(f"CLIP 模型加载完成: dim={self.dim}")

    def encode_image(self, image_path: str) -> np.ndarray:
        """编码单张图片为向量"""
        from PIL import Image
        img = Image.open(image_path)
        embedding = self.model.encode(img, normalize_embeddings=True)
        return embedding

    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """批量编码图片"""
        from PIL import Image
        images = [Image.open(p) for p in image_paths if os.path.exists(p)]
        logger.info(f"有效图片: {len(images)}/{len(image_paths)}")
        embeddings = self.model.encode(
            images, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
        )
        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        """编码文本查询为向量"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """批量编码文本"""
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
