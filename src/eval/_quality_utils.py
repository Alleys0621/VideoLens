"""评估模块公共工具: 字段完整性 + 枚举合规检查.

供 stage3_build_quality (P1/P2 层) 和 stage3_kb_quality (P3-P5 层) 共享,
避免两份重复实现。
"""

from __future__ import annotations


def _nonempty(v) -> bool:
    """字段非空判定: None / 空串 / 空列表视为空.

    字符串要求 strip 后非空; 列表要求 len > 0; 其他类型只要非 None 即视为非空.
    """
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, list):
        return len(v) > 0
    return True


def _field_completeness(records: list[dict], required: list[str]) -> dict:
    """计算 records 每个必填字段的非空比例.

    Returns:
        {field: ratio}, ratio 范围 [0, 1], 保留 4 位小数. records 为空时全 0.
    """
    if not records:
        return {f: 0.0 for f in required}
    return {
        f: round(sum(1 for r in records if _nonempty(r.get(f))) / len(records), 4)
        for f in required
    }


def _enum_violations(records: list[dict], field: str, enum: set) -> list[dict]:
    """检查 records[][field] 是否落在 enum 内, 返回违规列表.

    跳过空值 (只检查有值但不在 enum 的). id 字段自动尝试 arc_id / character_id / event_id.
    """
    viols = []
    for r in records:
        v = r.get(field)
        if v and v not in enum:
            viols.append({
                "id": r.get("arc_id") or r.get("character_id") or r.get("event_id") or "?",
                "field": field,
                "value": v,
            })
    return viols
