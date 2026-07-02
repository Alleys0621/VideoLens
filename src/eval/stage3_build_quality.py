"""Stage 3 建库质量评估 — 字段完整性 + Schema 合规 + 软质量指标.

评估对象: scripts/stage3_p1p2.py --save 产出的 stage3_dryrun.json.

宽松校验规则 (用户拍板):
  Action 必填: actor / action / utterance (非空字符串)
  Event 必填: title / motivation / summary / retrieval_text (非空字符串)
  允许: target=[] / participants 含 char_unknown / outcome="无变化"

产出: data/output/{video}/build_quality_report.json
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

from src.core.config import get_config
from src.core.helpers.json_utils import load_json, save_json
from src.core.logging import get_logger
from src.eval._quality_utils import _field_completeness, _nonempty

logger = get_logger()

# 枚举约束
ACTION_ENUM = {
    "inform", "ask", "answer", "command", "refuse",
    "promise", "threaten", "deceive", "argue", "invite", "react",
}
MOTIVATION_CONFIDENCE_ENUM = {"explicit", "inferred"}

# 必填字段 (宽松)
ACTION_REQUIRED = ["actor", "action", "utterance"]
EVENT_REQUIRED = ["title", "motivation", "summary", "retrieval_text"]


def _is_empty_target(v: Any) -> bool:
    """target 视为空: "" / [] / [""] 这类."""
    if v is None or v == "":
        return True
    if isinstance(v, list):
        return len(v) == 0 or all(not _nonempty(x) for x in v)
    return False


def evaluate(stage3_data: dict) -> dict:
    """对 stage3_dryrun.json 内容做评估, 返回结构化报告 dict."""
    actions = stage3_data.get("actions", []) or []
    events = stage3_data.get("events", []) or []
    stats = stage3_data.get("stats", {}) or {}

    # ---------- 字段完整性 ----------
    action_field_completeness = _field_completeness(actions, ACTION_REQUIRED)
    event_field_completeness = _field_completeness(events, EVENT_REQUIRED)

    # ---------- Schema 合规 ----------
    schema_violations = []
    for a in actions:
        aid = a.get("action_id", "?")
        if a.get("action") not in ACTION_ENUM:
            schema_violations.append({
                "record_id": aid,
                "field": "action",
                "value": a.get("action"),
                "reason": "not_in_enum",
            })
        tgt = a.get("target")
        if not (isinstance(tgt, str) or isinstance(tgt, list)):
            schema_violations.append({
                "record_id": aid,
                "field": "target",
                "value": repr(tgt),
                "reason": "not_str_or_list",
            })

    for e in events:
        eid = e.get("event_id", "?")
        mc = e.get("motivation_confidence")
        if mc not in MOTIVATION_CONFIDENCE_ENUM:
            schema_violations.append({
                "record_id": eid,
                "field": "motivation_confidence",
                "value": mc,
                "reason": "not_in_enum",
            })
        for ea in e.get("actions", []) or []:
            tgt = ea.get("target")
            if not (isinstance(tgt, str) or isinstance(tgt, list)):
                schema_violations.append({
                    "record_id": eid,
                    "field": "actions[].target",
                    "value": repr(tgt),
                    "reason": "not_str_or_list",
                })

    # ---------- 软质量指标 ----------
    n_actions = len(actions)
    n_events = len(events)

    vp_low = sum(
        1 for a in actions
        if (a.get("evidence") or {}).get("vp_low_confidence") is True
    )
    char_unknown_events = sum(
        1 for e in events
        if any("unknown" in (p or "") for p in (e.get("participants") or []))
    )
    empty_target_actions = sum(1 for a in actions if _is_empty_target(a.get("target")))
    empty_target_links = sum(
        1 for e in events for ea in (e.get("actions") or [])
        if _is_empty_target(ea.get("target"))
    )
    total_links = sum(len(e.get("actions") or []) for e in events)
    events_with_empty_actions = sum(
        1 for e in events if not (e.get("actions") or [])
    )

    quality_soft_metrics = {
        "vp_low_confidence_ratio": round(vp_low / n_actions, 4) if n_actions else 0.0,
        "char_unknown_event_ratio": round(char_unknown_events / n_events, 4) if n_events else 0.0,
        "empty_target_action_ratio": round(empty_target_actions / n_actions, 4) if n_actions else 0.0,
        "empty_target_link_ratio": round(empty_target_links / total_links, 4) if total_links else 0.0,
        "event_with_empty_actions_ratio": round(events_with_empty_actions / n_events, 4) if n_events else 0.0,
    }

    # ---------- 覆盖率 ----------
    n_candidates = stats.get("n_candidates_after_dedup", 0) or n_actions
    coverage = {
        "actions_per_candidate": round(n_actions / n_candidates, 4) if n_candidates else 0.0,
        "events_per_action": round(n_events / n_actions, 4) if n_actions else 0.0,
        "avg_actions_per_event": round(total_links / n_events, 2) if n_events else 0.0,
    }

    # ---------- Action 类型分布 (从 stats 复用, 没有就重算) ----------
    action_type_dist = stats.get("action_type_dist") or dict(
        Counter(a.get("action", "?") for a in events)
    )

    return {
        "summary": {
            "n_actions": n_actions,
            "n_events": n_events,
            "n_schema_violations": len(schema_violations),
        },
        "action_field_completeness": action_field_completeness,
        "event_field_completeness": event_field_completeness,
        "schema_violations": schema_violations,
        "quality_soft_metrics": quality_soft_metrics,
        "coverage": coverage,
        "action_type_dist": action_type_dist,
        "motivation_confidence_dist": stats.get("motivation_confidence_dist", {}),
    }


def main(video_dir: str) -> dict:
    """对指定 video_dir 跑评估, 写报告, 返回报告 dict."""
    config = get_config()
    output_dir = os.path.join(config.output_root, video_dir)
    stage3_path = os.path.join(output_dir, "stage3_dryrun.json")
    if not os.path.isfile(stage3_path):
        raise FileNotFoundError(f"未找到 {stage3_path}; 先跑 scripts.stage3_p1p2 --save")

    stage3_data = load_json(stage3_path)
    report = evaluate(stage3_data)

    out_path = os.path.join(output_dir, "build_quality_report.json")
    save_json(report, out_path)
    logger.info(f"建库评估报告已保存: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3 建库质量评估")
    parser.add_argument("--video", default="家有儿女/第001集", help="video_dir")
    args = parser.parse_args()
    main(args.video)
