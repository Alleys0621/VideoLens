"""通义千问文本模型调用"""

from vl.core.llm.base_client import BaseLLMClient


class QwenTextClient(BaseLLMClient):
    """通义千问文本模型客户端"""

    def generate(
        self,
        prompt: str,
        system: str = "你是一个专业的影视内容分析师。",
        model: str = "",
        temperature: float = 0.1,
        stage: str = "",
    ) -> str | None:
        """生成文本回复"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, model=model, temperature=temperature, stage=stage)

    def generate_json(
        self,
        prompt: str,
        system: str = "你是一个专业的影视内容分析师。请以 JSON 格式输出。",
        model: str = "",
        stage: str = "",
    ) -> str | None:
        """生成 JSON 格式回复"""
        raw = self.generate(prompt, system=system, model=model, stage=stage)
        if raw:
            return self.extract_json(raw)
        return None
