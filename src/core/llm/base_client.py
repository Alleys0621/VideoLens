"""LLM 基础客户端 - OpenAI 兼容 API"""

import time

from openai import OpenAI

from src.core.config import get_config
from src.core.cost import get_cost_tracker
from src.core.helpers.text_utils import extract_json as _extract_json


def _report_usage(model: str, usage, latency: float, stage: str):
    """从 OpenAI 兼容 usage 对象拆分模态 token 后上报 CostTracker

    DashScope 的 omni/vl 模型会返回 prompt_tokens_details.{text,audio}_tokens
    与 completion_tokens_details.text_tokens；图片 token 通常不单独回传，
    用 (prompt_tokens - audio - text) 估算。
    纯文本模型不会返回 details，回退到 prompt_tokens / completion_tokens。
    """
    if usage is None:
        get_cost_tracker().record(model=model, latency=latency, stage=stage)
        return

    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
    completion_tok = getattr(usage, "completion_tokens", 0) or 0

    text_in = audio_in = 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        audio_in = getattr(ptd, "audio_tokens", None) or 0
        text_in = getattr(ptd, "text_tokens", None) or 0

    text_out = 0
    ctd = getattr(usage, "completion_tokens_details", None)
    if ctd is not None:
        text_out = getattr(ctd, "text_tokens", None) or 0

    # 文本输出回退：未返回 details 时使用总 completion_tokens
    if text_out == 0 and completion_tok:
        text_out = completion_tok

    # 文本输入回退：纯文本模型未返回 details 时使用总 prompt_tokens
    if text_in == 0 and audio_in == 0 and prompt_tok:
        text_in = prompt_tok

    # 图片输入估算：总输入 - 已知音频/文本 (vl 模型通常不回传 image_tokens)
    image_in = max(0, prompt_tok - text_in - audio_in) if (text_in or audio_in) else 0

    get_cost_tracker().record(
        model=model,
        text_tokens_in=text_in,
        audio_tokens_in=audio_in,
        image_tokens_in=image_in,
        text_tokens_out=text_out,
        latency=latency,
        stage=stage,
    )


class BaseLLMClient:
    """OpenAI 兼容 LLM 客户端基类"""

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        config = get_config()
        self.model = model or config.model_text
        self.client = OpenAI(
            api_key=api_key or config.dashscope_api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.1,
        stage: str = "",
        max_tokens: int | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """发送聊天请求，返回原始文本

        enable_thinking: qwen3.7 系列默认开启 thinking (思考过程 token),
        会让输出 token 暴涨 3-4 倍且耗时翻倍. 默认关闭, 仅在需要强推理时显式开启.
        """
        model = model or self.model
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # qwen3.7 系列通过 extra_body 关闭 thinking
        # 旧模型 (qwen-plus 等) 会忽略此参数, 无副作用
        kwargs["extra_body"] = {"enable_thinking": enable_thinking}
        t0 = time.time()
        response = self.client.chat.completions.create(**kwargs)
        latency = time.time() - t0

        _report_usage(model, response.usage, latency, stage)

        return response.choices[0].message.content

    def extract_json(self, raw_output: str) -> str | None:
        """从 LLM 输出中提取 JSON"""
        return _extract_json(raw_output)

    def load_prompt_template(self, prompt_path: str) -> str:
        """加载 prompt 模板文件"""
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
