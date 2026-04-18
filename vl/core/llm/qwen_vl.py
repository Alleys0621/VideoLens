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
    ) -> Optional[str]:
        """分析多张图片"""
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
            content = choices[0]["message"]["content"]
            if isinstance(content, list):
                return content[0]["text"]
            return str(content)
        except Exception as e:
            print(f"Qwen VL 调用失败: {e}")
            return None
