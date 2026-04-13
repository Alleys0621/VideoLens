"""VideoLens 入口点"""

# 在任何其他 import 之前设置 HuggingFace 镜像
import os
from pathlib import Path

_env_file = Path(__file__).resolve().parents[3] / ".env"
if _env_file.is_file():
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value

# 确保 HF 镜像
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if not os.getenv("HUGGINGFACE_HUB_URL"):
    os.environ["HUGGINGFACE_HUB_URL"] = os.environ["HF_ENDPOINT"]

from vl.app.cli import app

if __name__ == "__main__":
    app()
