"""基于结构化知识库的问答模块

加载 Stage 5 生成的知识库，调用 LLM 回答用户问题。
"""

import json
import os

from vl.core.config import AppConfig
from vl.core.helpers.json_utils import load_json
from vl.core.logging import get_logger

logger = get_logger()


def load_knowledge_base(video_id: str, output_dir: str) -> dict:
    """
    加载视频的结构化知识库。

    Returns:
        knowledge_base dict, 包含 video_title -> phases -> events 结构
    """
    kb_path = os.path.join(output_dir, "stage5_knowledge", video_id, "knowledge_base.json")
    if not os.path.isfile(kb_path):
        raise FileNotFoundError(f"知识库不存在: {kb_path}")
    return load_json(kb_path)


def answer_question(
    question: str,
    video_id: str,
    output_dir: str,
    config: AppConfig,
) -> str:
    """
    基于知识库回答问题。

    Args:
        question: 用户问题
        video_id: 视频 ID
        output_dir: 输出根目录
        config: 应用配置

    Returns:
        LLM 回答文本
    """
    kb = load_knowledge_base(video_id, output_dir)

    # 取 video_title (KB 的顶层 key)
    video_title = list(kb.keys())[0] if kb else video_id

    # 将知识库转为紧凑文本作为上下文
    kb_text = _format_knowledge_base(kb)

    # 加载 prompt
    prompts = config.prompts.get("kb_qa", {})
    user_template = prompts.get("user", "")
    system_prompt = prompts.get("system", "")

    if not user_template:
        raise ValueError("kb_qa prompt 未配置，请检查 config/prompts.yaml")

    prompt = user_template.format(
        video_title=video_title,
        knowledge_base=kb_text,
        question=question,
    )

    # 调用 LLM
    from vl.core.llm.qwen_text import QwenTextClient

    client = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
    answer = client.generate(prompt, system=system_prompt)

    return answer or "(LLM 未返回结果)"


def batch_answer(
    qa_pairs: list[dict],
    video_id: str,
    output_dir: str,
    config: AppConfig,
) -> list[dict]:
    """
    批量回答问题。

    Args:
        qa_pairs: [{"question": "...", "answer": "参考答案"}]

    Returns:
        results: [{question, reference, answer, error}]
    """
    kb = load_knowledge_base(video_id, output_dir)
    video_title = list(kb.keys())[0] if kb else video_id
    kb_text = _format_knowledge_base(kb)

    prompts = config.prompts.get("kb_qa", {})
    user_template = prompts.get("user", "")
    system_prompt = prompts.get("system", "")

    if not user_template:
        raise ValueError("kb_qa prompt 未配置")

    from vl.core.llm.qwen_text import QwenTextClient
    client = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

    results = []
    total = len(qa_pairs)
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        reference = qa.get("answer", "")
        logger.info("回答 %d/%d: %s", i + 1, total, question[:50])

        prompt = user_template.format(
            video_title=video_title,
            knowledge_base=kb_text,
            question=question,
        )

        result = {"question": question, "reference": reference, "answer": "", "error": ""}
        try:
            answer = client.generate(prompt, system=system_prompt)
            result["answer"] = answer or "(LLM 未返回结果)"
        except Exception as e:
            result["error"] = str(e)
            logger.warning("问题 %d 回答失败: %s", i + 1, e)

        results.append(result)

    return results


def _format_knowledge_base(kb: dict) -> str:
    """将知识库 dict 格式化为 LLM 友好的文本"""
    lines = []
    for video_title, phases in kb.items():
        lines.append(f"【{video_title}】")
        for phase_name, events in phases.items():
            lines.append(f"  {phase_name}:")
            for eid, edata in events.items():
                if isinstance(edata, dict):
                    chars = "、".join(edata.get("characters", []))
                    action = edata.get("action", "")
                    goal = edata.get("goal", "")
                    summary = edata.get("summary", "")
                    lines.append(
                        f"    {eid}: [{chars}] {action} | 目标: {goal} | {summary}"
                    )
                else:
                    lines.append(f"    {eid}: {edata}")
    return "\n".join(lines)
