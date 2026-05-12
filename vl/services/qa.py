"""视频问答服务"""

import json

from vl.core.config import AppConfig
from vl.core.helpers.prompt_loader import load_prompt
from vl.core.logging import get_logger
from vl.services.search import search_scenes

logger = get_logger()


def answer_question(
    question: str,
    video: str = "",
    top_k: int = 5,
) -> tuple[str, list[dict]]:
    """基于检索到的场景上下文回答问题。

    Returns:
        (answer, reference_scenes)
    """
    results, config = search_scenes(question, video, top_k)

    if results is None or not results:
        raise FileNotFoundError("没有找到相关场景。请先运行 'videolens index' 建立索引。")

    if not config.dashscope_api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY，无法进行问答。")

    # 拼接上下文
    context_parts = []
    for i, r in enumerate(results, 1):
        start = r.get("start_time", 0)
        end = r.get("end_time", 0)
        part = f"[场景 {i}] {start:.1f}s - {end:.1f}s"
        if r.get("structured_caption"):
            part += f"\n  视觉: {json.dumps(r['structured_caption'], ensure_ascii=False)}"
        if r.get("vlm_caption"):
            part += f"\n  描述: {r['vlm_caption'][:300]}"
        if r.get("transcript"):
            part += f"\n  台词: {r['transcript'][:300]}"
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    # 调用 LLM
    from vl.core.llm.qwen_text import QwenTextClient
    qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

    user_tpl, sys_prompt = load_prompt(config, "qa_answer")
    if not user_tpl:
        raise ValueError("qa_answer prompt 未配置，请检查 config/prompts.yaml")

    prompt = user_tpl.format(question=question, context=context)

    logger.info("正在生成回答...")
    answer = qwen.generate(prompt, system=sys_prompt)

    return answer or "(LLM 未返回结果)", results
