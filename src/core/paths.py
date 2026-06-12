"""路径管理器 - 统一管理所有输入输出路径"""

import os
from src.core.config import get_config


class PathManager:
    """集中管理项目路径，自动创建目录"""

    def __init__(self, video_id: str = ""):
        self.config = get_config()
        self.video_id = video_id
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保关键目录存在"""
        for d in [
            self.stage1_dir,
            self.stage2_dir,
            self.stage3_dir,
            self.stage4_dir,
            self.stage5_dir,
            self.checkpoints_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        if self.video_id:
            for d in [
                self.video_stage1_dir,
                self.video_keyframes_dir,
                self.video_stage2_dir,
                self.video_stage3_dir,
                self.video_stage4_dir,
                self.video_stage5_dir,
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

    # --- Stage output directories ---

    @property
    def stage1_dir(self) -> str:
        return os.path.join(self.output_root, "stage1_scenes")

    @property
    def stage2_dir(self) -> str:
        return os.path.join(self.output_root, "stage2_features")

    @property
    def stage3_dir(self) -> str:
        return os.path.join(self.output_root, "stage3_captions")

    @property
    def stage4_dir(self) -> str:
        return os.path.join(self.output_root, "stage4_events")

    @property
    def stage5_dir(self) -> str:
        return os.path.join(self.output_root, "stage5_knowledge")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.output_root, "checkpoints")

    # --- Video-specific stage directories ---

    @property
    def video_stage1_dir(self) -> str:
        return os.path.join(self.stage1_dir, self.video_id)

    @property
    def video_keyframes_dir(self) -> str:
        return os.path.join(self.video_stage1_dir, "keyframes")

    @property
    def video_stage2_dir(self) -> str:
        return os.path.join(self.stage2_dir, self.video_id)

    @property
    def video_stage3_dir(self) -> str:
        return os.path.join(self.stage3_dir, self.video_id)

    @property
    def video_stage4_dir(self) -> str:
        return os.path.join(self.stage4_dir, self.video_id)

    @property
    def video_stage5_dir(self) -> str:
        return os.path.join(self.stage5_dir, self.video_id)

    # --- File paths ---

    @property
    def scenes_json_path(self) -> str:
        return os.path.join(self.video_stage1_dir, "scenes.json")

    @property
    def metadata_json_path(self) -> str:
        return os.path.join(self.video_stage1_dir, "metadata.json")

    @property
    def audio_path(self) -> str:
        return os.path.join(self.stage2_dir, "preprocessing", f"{self.video_id}.wav")

    @property
    def transcript_json_path(self) -> str:
        return os.path.join(self.video_stage2_dir, "transcript.json")

    @property
    def characters_json_path(self) -> str:
        return os.path.join(self.video_stage2_dir, "characters.json")

    @property
    def voiceprint_result_path(self) -> str:
        return os.path.join(self.video_stage2_dir, "voiceprint_result.json")

    @property
    def clip_vectors_path(self) -> str:
        return os.path.join(self.video_stage2_dir, "clip_vectors.npy")

    @property
    def clip_vectors_path_exists(self) -> bool:
        return os.path.isfile(self.clip_vectors_path)

    @property
    def captions_json_path(self) -> str:
        return os.path.join(self.video_stage3_dir, "captions.json")

    @property
    def faiss_index_path(self) -> str:
        return os.path.join(self.video_stage3_dir, "index.faiss")

    @property
    def doc_store_path(self) -> str:
        return os.path.join(self.video_stage3_dir, "doc_store.json")

    @property
    def events_json_path(self) -> str:
        return os.path.join(self.video_stage4_dir, "events.json")

    @property
    def knowledge_base_path(self) -> str:
        return os.path.join(self.video_stage5_dir, "knowledge_base.json")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoints_dir, f"{self.video_id}.json")

    def keyframe_path(self, scene_index: int) -> str:
        return os.path.join(
            self.video_keyframes_dir, f"scene_{scene_index:04d}.jpg"
        )
