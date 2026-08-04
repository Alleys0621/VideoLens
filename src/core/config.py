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
    # LLM 接入点 (默认经典 DashScope 公共端点; 用 Bailian 专属应用时改为专属域名)
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 意图路由专用模型 (轻量, 可单独换 qwen-flash / glm-4-flash 等)
    model_intent: str = "qwen3.7-flash"

    # Models
    model_omni_plus: str = "qwen3.5-omni-plus"
    model_omni_flash: str = "qwen3.5-omni-flash"
    model_vlm: str = "qwen-vl-max"
    model_text: str = "qwen3.7-plus"
    model_text_flash: str = "qwen3.7-flash"  # 主回复 LLM 高置信分支
    model_ocr: str = "qwen3-vl-plus"  # 字幕 OCR 专用 (输入便宜, 适合多帧采样)
    model_embedding: str = "qwen3.7-text-embedding"

    # Routing (意图路由 + 主 LLM 切换)
    intent_mode: str = "hybrid"
    hybrid_threshold: float = 0.65
    flash_threshold: float = 0.75

    # 调试 / 运行时开关
    perf_enabled: bool = False             # VIDEOLENS_PERF: 模块级耗时打点进 payload
    warmup_disabled: bool = False          # DISABLE_WARMUP: 跳过启动预热

    # 外部凭证 (从 .env 读, 不进 pipeline.yaml)
    openai_api_key: str = ""               # Mem0 / OpenAI 兼容客户端
    tavily_api_key: str = ""               # Web 搜索
    xfyun_app_id: str = ""
    xfyun_api_key: str = ""
    xfyun_api_secret: str = ""
    postgres_url: str = ""                 # 业务库连接串

    # Stage 2: 视觉处理 (Stage 1 参数全部通过 CLI --chunk / 硬编码传入, 不进 config)
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
    s2 = pipeline.get("stage2", {})
    paths_cfg = pipeline.get("paths", {})
    pricing_cfg = pipeline.get("pricing", {})
    routing_cfg = pipeline.get("routing", {})

    data_root = os.path.join(project_root, paths_cfg.get("data_root", "data"))
    output_root = os.path.join(project_root, paths_cfg.get("output_root", "data/output"))

    return AppConfig(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        dashscope_base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        model_intent=models.get("intent", "qwen3.7-flash"),
        model_omni_plus=models.get("omni_plus", "qwen3.5-omni-plus"),
        model_omni_flash=models.get("omni_flash", "qwen3.5-omni-flash"),
        model_vlm=models.get("vlm", "qwen-vl-max"),
        model_text=models.get("text", "qwen3.7-plus"),
        model_text_flash=models.get("text_flash", "qwen3.7-flash"),
        model_ocr=models.get("ocr", "qwen3-vl-plus"),
        model_embedding=models.get("embedding", "qwen3.7-text-embedding"),
        intent_mode=routing_cfg.get("intent_mode", os.getenv("INTENT_MODE", "hybrid")),
        hybrid_threshold=float(routing_cfg.get("hybrid_threshold", os.getenv("INTENT_HYBRID_THRESHOLD", "0.65"))),
        flash_threshold=float(routing_cfg.get("flash_threshold", os.getenv("MAIN_LLM_FLASH_THRESHOLD", "0.75"))),
        perf_enabled=os.getenv("VIDEOLENS_PERF", "0") == "1",
        warmup_disabled=os.getenv("DISABLE_WARMUP", "0") == "1",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        xfyun_app_id=os.getenv("XFYUN_APP_ID", ""),
        xfyun_api_key=os.getenv("XFYUN_API_KEY", ""),
        xfyun_api_secret=os.getenv("XFYUN_API_SECRET", ""),
        postgres_url=(
            os.getenv("POSTGRES_URL")
            or os.getenv("DATABASE_URI")
            or "postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens"
        ),
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
