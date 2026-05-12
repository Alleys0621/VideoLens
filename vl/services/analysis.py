"""视频分析服务"""

import json
import os

from vl.core.config import get_config
from vl.core.helpers.json_utils import load_json
from vl.core.helpers.prompt_loader import load_prompt
from vl.core.logging import get_logger

logger = get_logger()


def analyze_video(
    video_id: str,
    analysis_type: str = "summary",
) -> str:
    """生成视频分析 (摘要/角色/时间线)。

    Returns:
        分析结果文本
    """
    config = get_config()

    doc_path = os.path.join(config.output_root, "stage3_captions", video_id, "doc_store.json")
    if not os.path.isfile(doc_path):
        raise FileNotFoundError(f"未找到视频 {video_id} 的索引数据。请先运行 'videolens index'。")

    docs = load_json(doc_path)

    # 拼接上下文
    context_parts = []
    for doc in docs:
        start = doc.get("start_time", 0)
        end = doc.get("end_time", 0)
        part = f"[{start:.1f}s - {end:.1f}s]"
        if doc.get("structured_caption"):
            part += f" {json.dumps(doc['structured_caption'], ensure_ascii=False)}"
        if doc.get("transcript"):
            part += f" 台词: {doc['transcript'][:200]}"
        context_parts.append(part)

    context = "\n".join(context_parts)

    if not config.dashscope_api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY，无法进行分析。")

    user_tpl, sys_prompt = load_prompt(config, f"analyze_{analysis_type}")
    if not user_tpl:
        raise ValueError(f"analyze_{analysis_type} prompt 未配置，请检查 config/prompts.yaml")

    from vl.core.llm.qwen_text import QwenTextClient
    qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

    prompt = user_tpl.format(
        video_id=video_id,
        scene_count=len(docs),
        context=context,
    )

    logger.info(f"正在生成{analysis_type}分析...")
    result = qwen.generate(prompt, system=sys_prompt)

    return result or "(未能生成分析结果)"
