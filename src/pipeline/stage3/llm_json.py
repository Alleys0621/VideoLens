"""Stage 3 LLM JSON 调用统一封装.

合并原 scripts/stage3_dryrun.py 和 stage3_full.py 各自的 JSON 解析 + 重试逻辑。
提供:
  - parse_json_with_repair: 容忍 ```json``` 包裹 + 闭合大括号修复
  - call_llm_json: 通用 LLM 调用 + 重试 + JSON 解析 (P2/P3/P4/P5 使用)

P1 因有"results 数量必须等于 batch 大小"的特殊校验, 在 p1_p2_actions.py
里保留独立重试循环, 但复用本模块的 parse_json_with_repair.
"""

from __future__ import annotations

from src.core.helpers.text_utils import extract_json_obj


def _try_repair_json_braces(raw: str) -> str | None:
    """JSON 闭合修复: 统计 { } 数量差异, 在末尾补全缺失的 } 让 json.loads 通过.

    仅当原文本以 { 开头时尝试。LLM 偶发少输出一个闭合大括号, 这是最常见的修复场景。
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None
    open_count = text.count("{")
    close_count = text.count("}")
    deficit = open_count - close_count
    if deficit <= 0:
        return text
    # 在末尾剥掉可能的尾随逗号/空白, 再补 deficit 个 }
    stripped = text.rstrip().rstrip(",").rstrip()
    return stripped + ("}" * deficit)


def parse_json_with_repair(raw: str):
    """先 extract_json_obj, 失败则尝试闭合修复后再解析。返回 None 表示彻底失败."""
    parsed = extract_json_obj(raw)
    if parsed is not None:
        return parsed
    repaired = _try_repair_json_braces(raw)
    if repaired is None:
        return None
    return extract_json_obj(repaired)


def call_llm_json(
    client,
    prompt: str,
    stage: str,
    max_tokens: int = 6000,
    max_retries: int = 2,
    expected_key: str | None = None,
    enable_thinking: bool = False,
):
    """调用 LLM 返回解析后的 JSON. 失败重试最多 max_retries 次.

    Args:
        client: QwenTextClient 实例
        prompt: 完整 prompt 字符串 (调用方负责模板替换)
        stage: cost 归属标签
        max_tokens: 输出上限
        max_retries: 失败重试次数
        expected_key: 若给定且 parsed 是 dict 含此 key, 返回 parsed[key] (如 "arc_updates")
        enable_thinking: 是否开启 qwen thinking 模式

    Returns:
        解析后的 JSON 结构 (dict / list / 标量). 若 expected_key 不存在则返回整个 parsed.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            raw = client.generate(
                prompt=prompt, stage=stage, max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            if not raw:
                raise RuntimeError("LLM 返回空")
            parsed = parse_json_with_repair(raw)
            if parsed is None:
                raise RuntimeError(f"返回非 JSON, 前 200 字: {raw[:200]!r}")
            if expected_key and isinstance(parsed, dict) and expected_key in parsed:
                return parsed[expected_key]
            return parsed
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                print(f"          [RETRY {attempt+1}/{max_retries}] {e}", flush=True)
    raise last_err
