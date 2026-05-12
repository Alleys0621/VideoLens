"""文本工具 - JSON 提取 (合并自 BaseLLMClient 和 QwenOmni)"""

import json
from typing import Optional

import regex


def extract_json(raw_output: str) -> Optional[str]:
    """从文本中提取 JSON 字符串。

    尝试策略:
      1. 直接解析 (如果文本本身是 JSON)
      2. 从 ```json ... ``` 代码块提取
      3. 递归正则匹配最外层 {...}
    """
    text = raw_output.strip()
    if not text:
        return None

    # 策略 1: 直接解析
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    # 策略 2: JSON 代码块
    match = regex.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, regex.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    # 策略 3: 递归正则匹配最外层 JSON 对象
    match = regex.search(r"\{(?:[^{}]|(?R))*\}", text, regex.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    return None


def extract_json_obj(raw_output: str):
    """从文本中提取 JSON 并解析为 Python 对象。"""
    extracted = extract_json(raw_output)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    return None
