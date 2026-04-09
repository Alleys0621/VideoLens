"""LLM 基础客户端 - OpenAI 兼容 API + JSON 提取 (移植自 MoocAI BaseGenerator)"""

import json
from typing import Optional

import regex
from openai import OpenAI

from vl_core.config import get_config


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
    ) -> str:
        """发送聊天请求，返回原始文本"""
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def extract_json(self, raw_output: str) -> Optional[str]:
        """从 LLM 输出中提取 JSON (使用递归正则匹配)"""
        match = regex.search(r"\{(?:[^{}]|(?R))*\}", raw_output, regex.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError as e:
                print(f"JSON 解析出错: {e}")
                return None
        print("未匹配到 JSON 结构")
        return None

    def load_prompt_template(self, prompt_path: str) -> str:
        """加载 prompt 模板文件"""
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
