"""LLM 基础客户端 - OpenAI 兼容 API"""

import time

from openai import OpenAI

from vl.core.config import get_config
from vl.core.cost import get_cost_tracker
from vl.core.helpers.text_utils import extract_json as _extract_json


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
    ) -> str:
        """发送聊天请求，返回原始文本"""
        model = model or self.model
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        latency = time.time() - t0

        # 上报用量
        usage = response.usage
        tracker = get_cost_tracker()
        tracker.record(
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency=latency,
            stage=stage,
        )

        return response.choices[0].message.content

    def extract_json(self, raw_output: str) -> str | None:
        """从 LLM 输出中提取 JSON"""
        return _extract_json(raw_output)

    def load_prompt_template(self, prompt_path: str) -> str:
        """加载 prompt 模板文件"""
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
