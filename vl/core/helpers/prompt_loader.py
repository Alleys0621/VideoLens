"""Prompt 加载工具"""

from vl.core.config import AppConfig


def load_prompt(config: AppConfig, key: str) -> tuple[str, str]:
    """从 prompts.yaml 加载 prompt 模板。

    Args:
        config: 应用配置
        key: prompt 键名 (如 "stage2_omni", "scene_caption")

    Returns:
        (user_template, system_prompt)
    """
    prompts = config.prompts.get(key, {})
    return prompts.get("user", ""), prompts.get("system", "")
