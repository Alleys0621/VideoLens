"""VideoLens 配置管理 - 从 .env 和 pipeline.yaml 加载配置"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _find_project_root() -> str:
    """从当前文件向上查找项目根目录（包含 pyproject.toml 且 name = videolens）"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if 'name = "videolens"' in content:
                    return str(parent)
            except (OSError, UnicodeDecodeError):
                pass
    raise RuntimeError("未找到项目根目录")


@dataclass(frozen=True)
class AppConfig:
    """应用全局配置"""
    # API Keys
    dashscope_api_key: str = ""

    # Models
    model_vlm: str = "qwen-vl-max"
    model_text: str = "qwen-plus"
    model_omni: str = "qwen3.5-omni-plus"
    model_clip: str = "sentence-transformers/clip-ViT-B-32"

    # Scene Detection
    scene_backend: str = "pyscenedetect"
    content_threshold: float = 27.0
    min_scene_len: float = 1.0
    samples_per_scene: int = 2       # 每个场景提取的关键帧数

    # VLM Sliding Window
    vlm_window_size: int = 4         # 滑动窗口大小 (帧数)
    vlm_stride: int = 2              # 滑动窗口步长

    # ASR (qwen-omni)
    asr_language: str = "zh"
    asr_chunk_duration: int = 120
    asr_max_keyframes_per_chunk: int = 5

    # Voiceprint (讯飞声纹识别)
    voiceprint_enabled: bool = False
    voiceprint_app_id: str = ""
    voiceprint_api_key: str = ""
    voiceprint_api_secret: str = ""
    voiceprint_group_id: str = "default_group"
    voiceprint_score_threshold: float = 0.3
    voiceprint_min_duration: float = 3.0
    voiceprint_name_mapping: dict = field(default_factory=dict)

    # CLIP
    clip_embedding_dim: int = 512
    clip_batch_size: int = 32

    # Retrieval
    retrieval_top_k: int = 10
    retrieval_rerank: bool = True

    # Paths
    project_root: str = ""
    data_root: str = ""
    output_root: str = ""

    # Prompts
    prompts: dict = field(default_factory=dict)

    # Pricing
    pricing: dict = field(default_factory=dict)


def load_config() -> AppConfig:
    """加载完整配置"""
    project_root = _find_project_root()

    # Load .env (DASHSCOPE_API_KEY etc.)
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path, override=False)

    # Load pipeline.yaml
    pipeline_path = os.path.join(project_root, "config", "pipeline.yaml")
    pipeline = {}
    if os.path.isfile(pipeline_path):
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline = yaml.safe_load(f) or {}

    # Load prompts.yaml
    prompts_path = os.path.join(project_root, "config", "prompts.yaml")
    prompts = {}
    if os.path.isfile(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f) or {}

    models = pipeline.get("models", {})
    scene_cfg = pipeline.get("scene_detection", {})
    asr_cfg = pipeline.get("asr", {})
    clip_cfg = pipeline.get("clip", {})
    retrieval_cfg = pipeline.get("retrieval", {})
    paths_cfg = pipeline.get("paths", {})
    pricing_cfg = pipeline.get("pricing", {})
    vp_cfg = pipeline.get("voiceprint", {})

    data_root = os.path.join(project_root, paths_cfg.get("data_root", "data"))
    output_root = os.path.join(project_root, paths_cfg.get("output_root", "data/output"))

    return AppConfig(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        model_vlm=models.get("vlm", "qwen-vl-max"),
        model_text=models.get("text", "qwen-plus"),
        model_omni=models.get("omni", "qwen3.5-omni-plus"),
        model_clip=models.get("clip", "sentence-transformers/clip-ViT-B-32"),
        scene_backend=scene_cfg.get("backend", "pyscenedetect"),
        content_threshold=scene_cfg.get("content_threshold", 27.0),
        min_scene_len=scene_cfg.get("min_scene_len", 1.0),
        samples_per_scene=scene_cfg.get("samples_per_scene", 8),
        vlm_window_size=scene_cfg.get("vlm_window_size", 4),
        vlm_stride=scene_cfg.get("vlm_stride", 2),
        asr_language=asr_cfg.get("language", "zh"),
        asr_chunk_duration=asr_cfg.get("chunk_duration", 120),
        asr_max_keyframes_per_chunk=asr_cfg.get("max_keyframes_per_chunk", 5),
        clip_embedding_dim=clip_cfg.get("embedding_dim", 512),
        clip_batch_size=clip_cfg.get("batch_size", 32),
        retrieval_top_k=retrieval_cfg.get("top_k", 10),
        retrieval_rerank=retrieval_cfg.get("use_llm_rerank", True),
        voiceprint_enabled=vp_cfg.get("enabled", False),
        voiceprint_app_id=os.getenv("XFYUN_APP_ID", ""),
        voiceprint_api_key=os.getenv("XFYUN_API_KEY", ""),
        voiceprint_api_secret=os.getenv("XFYUN_API_SECRET", ""),
        voiceprint_group_id=vp_cfg.get("group_id", "default_group"),
        voiceprint_score_threshold=vp_cfg.get("score_threshold", 0.3),
        voiceprint_min_duration=vp_cfg.get("min_duration", 3.0),
        voiceprint_name_mapping=vp_cfg.get("name_mapping", {}),
        project_root=project_root,
        data_root=data_root,
        output_root=output_root,
        prompts=prompts,
        pricing=pricing_cfg,
    )


# Module-level singleton
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
