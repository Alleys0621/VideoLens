"""通义千问 VL 多模态调用 (OpenAI 兼容接口)"""

import base64
import time

from openai import OpenAI

from src.core.config import get_config
from src.core.cost import get_cost_tracker
from src.core.logging import get_logger

logger = get_logger()


class QwenVLClient:
    """通义千问视觉语言模型客户端"""

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        config = get_config()
        self.model = model or config.model_vlm
        self.api_key = api_key or config.dashscope_api_key
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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

    def _encode_image(self, image_path: str) -> str:
        """读取图片并 base64 编码"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _call(self, image_paths: list[str], prompt: str, stage: str = "") -> str | None:
        """单次 VLM 调用"""
        content = []
        for path in image_paths:
            b64 = self._encode_image(path)
            ext = path.rsplit(".", 1)[-1].lower()
            mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                stream=True,
                stream_options={"include_usage": True},
            )

            full_text = ""
            in_tok = out_tok = 0
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage:
                    in_tok = chunk.usage.prompt_tokens or 0
                    out_tok = chunk.usage.completion_tokens or 0

            latency = time.time() - t0
            # Qwen-VL API 返回的 prompt_tokens 已合并文本+图片 tokens;
            # qwen-vl-max/plus 的 text_input 与 image_input 单价相同, 统一计入 text_tokens_in.
            get_cost_tracker().record(
                model=self.model,
                text_tokens_in=in_tok,
                text_tokens_out=out_tok,
                latency=latency,
                stage=stage,
            )
            return full_text
        except Exception as e:
            latency = time.time() - t0
            get_cost_tracker().record(model=self.model, latency=latency, stage=stage)
            logger.error(f"Qwen VL 调用失败: {e}")
            return None
