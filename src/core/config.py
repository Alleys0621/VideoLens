"""VideoLens 配置管理"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _find_project_root() -> str:
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
    dashscope_api_key: str = ""

    # Models
    model_omni_plus: str = "qwen3.5-omni-plus"
    model_omni_flash: str = "qwen3.5-omni-flash"
    model_vlm: str = "qwen-vl-max"
    model_text: str = "qwen-plus"

    # Stage 1: 音频处理
    theme_window: int = 120
    chunk_duration: int = 60
    silence_db: float = -30.0
    silence_remove_min: float = 0.3
    speech_remove_max: float = 0.3
    name_mapping: dict = field(default_factory=dict)

    # Stage 2: 视觉处理
    content_threshold: float = 27.0
    min_scene_len: float = 1.0
    samples_per_scene: int = 8

    # Paths
    project_root: str = ""
    data_root: str = ""
    output_root: str = ""

    # Prompts & Pricing
    prompts: dict = field(default_factory=dict)
    pricing: dict = field(default_factory=dict)


def load_config() -> AppConfig:
    project_root = _find_project_root()

    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path, override=False)

    pipeline_path = os.path.join(project_root, "config", "pipeline.yaml")
    pipeline = {}
    if os.path.isfile(pipeline_path):
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline = yaml.safe_load(f) or {}

    prompts_path = os.path.join(project_root, "config", "prompts.yaml")
    prompts = {}
    if os.path.isfile(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f) or {}

    models = pipeline.get("models", {})
    s1 = pipeline.get("stage1", {})
    s2 = pipeline.get("stage2", {})
    paths_cfg = pipeline.get("paths", {})
    pricing_cfg = pipeline.get("pricing", {})

    data_root = os.path.join(project_root, paths_cfg.get("data_root", "data"))
    output_root = os.path.join(project_root, paths_cfg.get("output_root", "data/output"))

    return AppConfig(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        model_omni_plus=models.get("omni_plus", "qwen3.5-omni-plus"),
        model_omni_flash=models.get("omni_flash", "qwen3.5-omni-flash"),
        model_vlm=models.get("vlm", "qwen-vl-max"),
        model_text=models.get("text", "qwen-plus"),
        theme_window=s1.get("theme_window", 120),
        chunk_duration=s1.get("chunk_duration", 60),
        silence_db=s1.get("silence_db", -30.0),
        silence_remove_min=s1.get("silence_remove_min", 0.3),
        speech_remove_max=s1.get("speech_remove_max", 0.3),
        name_mapping=s1.get("name_mapping", {}),
        content_threshold=s2.get("content_threshold", 27.0),
        min_scene_len=s2.get("min_scene_len", 1.0),
        samples_per_scene=s2.get("samples_per_scene", 8),
        project_root=project_root,
        data_root=data_root,
        output_root=output_root,
        prompts=prompts,
        pricing=pricing_cfg,
    )


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
