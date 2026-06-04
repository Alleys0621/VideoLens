"""通义千问 VL 多模态调用"""

import time

from dashscope import MultiModalConversation

from vl.core.config import get_config
from vl.core.cost import get_cost_tracker
from vl.core.logging import get_logger

logger = get_logger()


class QwenVLClient:
    """通义千问视觉语言模型客户端"""

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        config = get_config()
        self.model = model or config.model_vlm
        self.api_key = api_key or config.dashscope_api_key
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        stage: str = "",
    ) -> str | None:
        """分析单张图片"""
        return self.analyze_images([image_path], prompt, stage=stage)

    def analyze_images(
        self,
        image_paths: list[str],
        prompt: str,
        window_size: int = 4,
        stride: int = 2,
        stage: str = "",
    ) -> str | None:
        """分析多张图片，使用滑动窗口处理。"""
        n = len(image_paths)
        if n == 0:
            return None
        if n <= window_size:
            return self._call(image_paths, prompt, stage=stage)

        prev_result = ""
        for start in range(0, n - window_size + 1, stride):
            window = image_paths[start:start + window_size]
            if prev_result:
                context_prompt = (
                    f"{prompt}\n\n"
                    f"【上一窗口分析结果（请在此基础上增量更新，修正错误，补充新信息）】\n"
                    f"{prev_result}"
                )
            else:
                context_prompt = prompt

            result = self._call(window, context_prompt, stage=stage)
            if result:
                prev_result = result

        return prev_result or None

    def _call(self, image_paths: list[str], prompt: str, stage: str = "") -> str | None:
        """单次 VLM 调用"""
        content = []
        for path in image_paths:
            content.append({"image": f"file://{path}"})
        content.append({"text": prompt})

        messages = [{"role": "user", "content": content}]

        t0 = time.time()
        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            latency = time.time() - t0

            # 提取 token 用量
            usage = response.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # 上报用量
            tracker = get_cost_tracker()
            tracker.record(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                stage=stage,
            )

            choices = response["output"]["choices"]
            resp_content = choices[0]["message"]["content"]
            if isinstance(resp_content, list):
                return resp_content[0]["text"]
            return str(resp_content)
        except Exception as e:
            latency = time.time() - t0
            # 即使失败也记录调用 (无 token 数据)
            tracker = get_cost_tracker()
            tracker.record(model=self.model, latency=latency, stage=stage)
            logger.error(f"Qwen VL 调用失败: {e}")
            return None
