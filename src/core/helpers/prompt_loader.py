"""Prompt 加载工具"""

from src.core.config import AppConfig


def load_prompt(config: AppConfig, key: str) -> tuple[str, str]:
    """从 prompts.yaml 加载 prompt 模板。

    Args:
        config: 应用配置
        key: prompt 键名 (如 "stage1a_theme_detection", "stage2_scene_caption")

    Returns:
        (user_template, system_prompt)
    """
    prompts = config.prompts.get(key, {})
    return prompts.get("user", ""), prompts.get("system", "")
