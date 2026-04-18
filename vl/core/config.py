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
    model_whisper: str = "large-v3"
    model_clip: str = "sentence-transformers/clip-ViT-B-32"

    # Scene Detection
    scene_backend: str = "pyscenedetect"
    content_threshold: float = 27.0
    min_scene_len: float = 1.0

    # ASR
    asr_backend: str = "qwen-omni"  # "qwen-omni" / "qwen" / "whisper"
    asr_language: str = "zh"
    asr_beam_size: int = 5
    asr_vad_filter: bool = True
    asr_chunk_duration: int = 120
    asr_silence_min_len: int = 500
    asr_silence_thresh: int = -40
    asr_max_keyframes_per_chunk: int = 5

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

    data_root = os.path.join(project_root, paths_cfg.get("data_root", "data"))
    output_root = os.path.join(project_root, paths_cfg.get("output_root", "data/output"))

    return AppConfig(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        model_vlm=models.get("vlm", "qwen-vl-max"),
        model_text=models.get("text", "qwen-plus"),
        model_omni=models.get("omni", "qwen3.5-omni-plus"),
        model_whisper=models.get("whisper", "large-v3"),
        model_clip=models.get("clip", "sentence-transformers/clip-ViT-B-32"),
        scene_backend=scene_cfg.get("backend", "pyscenedetect"),
        content_threshold=scene_cfg.get("content_threshold", 27.0),
        min_scene_len=scene_cfg.get("min_scene_len", 1.0),
        asr_backend=asr_cfg.get("backend", "whisper"),
        asr_language=asr_cfg.get("language", "zh"),
        asr_beam_size=asr_cfg.get("beam_size", 5),
        asr_vad_filter=asr_cfg.get("vad_filter", True),
        asr_chunk_duration=asr_cfg.get("chunk_duration", 120),
        asr_silence_min_len=asr_cfg.get("silence_min_len", 500),
        asr_silence_thresh=asr_cfg.get("silence_thresh", -40),
        asr_max_keyframes_per_chunk=asr_cfg.get("max_keyframes_per_chunk", 5),
        clip_embedding_dim=clip_cfg.get("embedding_dim", 512),
        clip_batch_size=clip_cfg.get("batch_size", 32),
        retrieval_top_k=retrieval_cfg.get("top_k", 10),
        retrieval_rerank=retrieval_cfg.get("use_llm_rerank", True),
        project_root=project_root,
        data_root=data_root,
        output_root=output_root,
        prompts=prompts,
    )


# Module-level singleton
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
