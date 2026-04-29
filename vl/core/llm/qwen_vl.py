"""通义千问 VL 多模态调用"""

from typing import Optional

from dashscope import MultiModalConversation

from vl.core.config import get_config


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
    ) -> Optional[str]:
        """分析单张图片"""
        return self.analyze_images([image_path], prompt)

    def analyze_images(
        self,
        image_paths: list[str],
        prompt: str,
        window_size: int = 4,
        stride: int = 2,
    ) -> Optional[str]:
        """
        分析多张图片，使用滑动窗口处理。

        当图片数 <= window_size 时，单次调用；
        否则按滑动窗口分批调用，每个窗口携带上一窗口结果作为上下文增量更新。

        Example (8 frames, window_size=4, stride=2):
          window 0: frames[0:4]
          window 1: frames[2:6] + prev result
          window 2: frames[4:8] + prev result
        """
        n = len(image_paths)
        if n == 0:
            return None
        if n <= window_size:
            return self._call(image_paths, prompt)

        # 滑动窗口增量分析
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

            result = self._call(window, context_prompt)
            if result:
                prev_result = result

        return prev_result or None

    def _call(self, image_paths: list[str], prompt: str) -> Optional[str]:
        """单次 VLM 调用"""
        content = []
        for path in image_paths:
            content.append({"image": f"file://{path}"})
        content.append({"text": prompt})

        messages = [{"role": "user", "content": content}]

        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            choices = response["output"]["choices"]
            resp_content = choices[0]["message"]["content"]
            if isinstance(resp_content, list):
                return resp_content[0]["text"]
            return str(resp_content)
        except Exception as e:
            print(f"Qwen VL 调用失败: {e}")
            return None
