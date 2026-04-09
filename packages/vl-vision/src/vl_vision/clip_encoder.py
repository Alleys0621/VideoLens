"""CLIP 语义编码器 - 生成场景和文本的语义向量"""

import numpy as np
from sentence_transformers import SentenceTransformer


class CLIPEncoder:
    """CLIP 图文语义编码器"""

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_image(self, image_path: str) -> np.ndarray:
        """编码单张图片为向量"""
        from PIL import Image
        img = Image.open(image_path)
        embedding = self.model.encode(img, normalize_embeddings=True)
        return embedding

    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """批量编码图片"""
        from PIL import Image
        images = [Image.open(p) for p in image_paths]
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
