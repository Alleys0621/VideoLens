"""路径管理器 - 统一管理所有输入输出路径"""

import os
from vl.core.config import get_config


class PathManager:
    """集中管理项目路径，自动创建目录"""

    def __init__(self, video_id: str = ""):
        self.config = get_config()
        self.video_id = video_id
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保关键目录存在"""
        for d in [
            self.scenes_dir,
            self.transcripts_dir,
            self.characters_dir,
            self.embeddings_dir,
            self.index_dir,
            self.captions_dir,
            self.alignment_dir,
            self.knowledge_dir,
            self.checkpoints_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        if self.video_id:
            for d in [
                self.video_scenes_dir,
                self.video_keyframes_dir,
                self.video_transcripts_dir,
                self.video_embeddings_dir,
            ]:
                os.makedirs(d, exist_ok=True)

    @property
    def project_root(self) -> str:
        return self.config.project_root

    @property
    def data_root(self) -> str:
        return self.config.data_root

    @property
    def output_root(self) -> str:
        return self.config.output_root

    # --- Global output directories ---

    @property
    def scenes_dir(self) -> str:
        return os.path.join(self.output_root, "scenes")

    @property
    def transcripts_dir(self) -> str:
        return os.path.join(self.output_root, "transcripts")

    @property
    def characters_dir(self) -> str:
        return os.path.join(self.output_root, "characters")

    @property
    def embeddings_dir(self) -> str:
        return os.path.join(self.output_root, "embeddings")

    @property
    def index_dir(self) -> str:
        return os.path.join(self.output_root, "index")

    @property
    def captions_dir(self) -> str:
        return os.path.join(self.output_root, "captions")

    @property
    def alignment_dir(self) -> str:
        return os.path.join(self.output_root, "alignment")

    @property
    def knowledge_dir(self) -> str:
        return os.path.join(self.output_root, "knowledge")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.output_root, "checkpoints")

    # --- Video-specific directories ---

    @property
    def video_scenes_dir(self) -> str:
        return os.path.join(self.scenes_dir, self.video_id)

    @property
    def video_keyframes_dir(self) -> str:
        return os.path.join(self.video_scenes_dir, "keyframes")

    @property
    def video_transcripts_dir(self) -> str:
        return os.path.join(self.transcripts_dir, self.video_id)

    @property
    def video_embeddings_dir(self) -> str:
        return os.path.join(self.embeddings_dir, self.video_id)

    @property
    def video_index_dir(self) -> str:
        return os.path.join(self.index_dir, self.video_id)

    # --- File paths ---

    @property
    def scenes_json_path(self) -> str:
        return os.path.join(self.video_scenes_dir, "scenes.json")

    @property
    def metadata_json_path(self) -> str:
        return os.path.join(self.video_scenes_dir, "metadata.json")

    @property
    def transcript_json_path(self) -> str:
        return os.path.join(self.video_transcripts_dir, "transcript.json")

    @property
    def audio_path(self) -> str:
        return os.path.join(self.output_root, "preprocessing", f"{self.video_id}.wav")

    @property
    def clip_vectors_path(self) -> str:
        return os.path.join(self.video_embeddings_dir, "clip_vectors.npy")

    @property
    def faiss_index_path(self) -> str:
        return os.path.join(self.video_index_dir, "index.faiss")

    @property
    def doc_store_path(self) -> str:
        return os.path.join(self.video_index_dir, "doc_store.json")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoints_dir, f"{self.video_id}.json")

    def keyframe_path(self, scene_index: int) -> str:
        return os.path.join(
            self.video_keyframes_dir, f"scene_{scene_index:04d}.jpg"
        )
