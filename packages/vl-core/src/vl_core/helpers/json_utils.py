"""JSON 工具 - 文件读写 + 数据持久化"""

import json
import os
from typing import Any


def save_json(data: Any, path: str, ensure_dir: bool = True) -> None:
    """保存数据为 JSON 文件"""
    if ensure_dir:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
