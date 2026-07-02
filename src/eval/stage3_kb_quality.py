"""Stage 3 完整知识库质量评估

输入:
  - data/output/{video}/stage3_kb.json (含 events, arc_updates, video_summary)
  - data/output/_global/characters.json
  - data/output/_global/global_arcs.json

输出: data/output/{video}/kb_quality_report.json
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_config
from src.core.helpers.json_utils import load_json, save_json
from src.eval._quality_utils import _enum_violations, _field_completeness

# 字段必填 (宽松)
EVENT_REQUIRED = ["event_id", "title", "motivation", "summary", "retrieval_text"]
ARC_REQUIRED = ["arc_id", "title", "type", "status", "summary", "related_event_ids"]
VIDEO_REQUIRED = ["video_id", "title", "episode_summary", "character_refs", "main_arcs"]
CHAR_REQUIRED = ["character_id", "name", "role", "description"]
GLOBAL_ARC_REQUIRED = ["arc_id", "title", "type", "status", "summary"]

# 枚举
ARC_TYPE_ENUM = {"character_growth", "relationship", "conflict", "mystery", "goal", "other"}
ARC_STATUS_ENUM = {"open", "progressing", "resolved", "suspended"}
CHAR_ROLE_ENUM = {"主角", "配角", "客串", "反派", "路人"}
RELATION_TYPE_ENUM = {"family", "friend", "rival", "mentor", "enemy", "other"}


def evaluate(video_dir: str) -> dict:
    cfg = get_config()
    ep_dir = os.path.join(cfg.output_root, video_dir)
    kb_path = os.path.join(ep_dir, "stage3_kb.json")
    if not os.path.isfile(kb_path):
        raise FileNotFoundError(kb_path)
    kb = load_json(kb_path)

    global_dir = os.path.join(cfg.output_root, "_global")
    chars = load_json(os.path.join(global_dir, "characters.json")) if os.path.isfile(os.path.join(global_dir, "characters.json")) else []
    arcs = load_json(os.path.join(global_dir, "global_arcs.json")) if os.path.isfile(os.path.join(global_dir, "global_arcs.json")) else []

    events = kb.get("events", [])
    arc_updates = kb.get("arc_updates", [])
    video_summary = kb.get("video_summary", {})

    # 字段完整率
    event_fc = _field_completeness(events, EVENT_REQUIRED)
    arc_fc = _field_completeness(arc_updates, ARC_REQUIRED)
    video_fc = _field_completeness([video_summary], VIDEO_REQUIRED) if video_summary else {}
    char_fc = _field_completeness(chars, CHAR_REQUIRED)
    garc_fc = _field_completeness(arcs, GLOBAL_ARC_REQUIRED)

    # Schema 合规
    schema_violations = []
    schema_violations.extend([{"layer": "arc_updates", **v} for v in _enum_violations(arc_updates, "type", ARC_TYPE_ENUM)])
    schema_violations.extend([{"layer": "arc_updates", **v} for v in _enum_violations(arc_updates, "status", ARC_STATUS_ENUM)])
    schema_violations.extend([{"layer": "global_arcs", **v} for v in _enum_violations(arcs, "type", ARC_TYPE_ENUM)])
    schema_violations.extend([{"layer": "global_arcs", **v} for v in _enum_violations(arcs, "status", ARC_STATUS_ENUM)])
    schema_violations.extend([{"layer": "characters", **v} for v in _enum_violations(chars, "role", CHAR_ROLE_ENUM)])
    for c in chars:
        for rel in c.get("relationships", []) or []:
            if rel.get("relation_type") and rel.get("relation_type") not in RELATION_TYPE_ENUM:
                schema_violations.append({"layer": "characters", "id": c.get("character_id"),
                                          "field": "relationships.relation_type",
                                          "value": rel.get("relation_type")})

    # 引用一致性
    event_ids = {e.get("event_id") for e in events}
    arc_dangling_refs = []
    for a in arc_updates:
        for eid in a.get("related_event_ids", []) or []:
            if eid not in event_ids:
                arc_dangling_refs.append({"arc_id": a.get("arc_id"), "dangling": eid})

    video_main_arcs_dangling = [a for a in video_summary.get("main_arcs", []) if a not in {x.get("arc_id") for x in arc_updates}]
    video_key_events_dangling = [e for e in video_summary.get("key_events", []) if e not in event_ids]
    video_char_refs_count = len(video_summary.get("character_refs", []))
    chars_matched = sum(1 for c in video_summary.get("character_refs", [])
                        if any(c == ch.get("name") or c in (ch.get("aliases") or []) or c == ch.get("character_id")
                               for ch in chars))

    # 角色覆盖率
    char_in_events = set()
    for e in events:
        for p in e.get("participants", []) or []:
            char_in_events.add(p)
    char_global_names = {c.get("name") for c in chars}

    return {
        "video_id": kb.get("video_id"),
        "summary": {
            "n_events": len(events),
            "n_arc_updates": len(arc_updates),
            "n_global_characters": len(chars),
            "n_global_arcs": len(arcs),
            "n_schema_violations": len(schema_violations),
        },
        "completeness": {
            "event": event_fc,
            "arc_update": arc_fc,
            "video_summary": video_fc,
            "character": char_fc,
            "global_arc": garc_fc,
        },
        "schema_violations": schema_violations,
        "reference_integrity": {
            "arc_dangling_event_refs": arc_dangling_refs,
            "video_main_arcs_dangling": video_main_arcs_dangling,
            "video_key_events_dangling": video_key_events_dangling,
            "video_char_refs": video_char_refs_count,
            "video_char_refs_matched_in_global": chars_matched,
        },
        "character_coverage": {
            "unique_participants_in_events": len(char_in_events),
            "covered_by_global_chars": sum(1 for p in char_in_events if p in char_global_names),
            "missing": sorted(char_in_events - char_global_names),
        },
        "arc_distribution": {
            "by_type": dict(Counter(a.get("type") for a in arc_updates)),
            "by_status": dict(Counter(a.get("status") for a in arc_updates)),
        },
        "cost_cny": kb.get("cost", {}).get("total_cost_cny", 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="家有儿女/第001集")
    args = parser.parse_args()
    report = evaluate(args.video)
    cfg = get_config()
    out_path = os.path.join(cfg.output_root, args.video, "kb_quality_report.json")
    save_json(report, out_path)
    print(f"KB quality report: {out_path}")

    # 打印关键指标
    print(f"\n=== {args.video} KB Quality ===")
    print(f"  events: {report['summary']['n_events']}")
    print(f"  arc_updates: {report['summary']['n_arc_updates']}")
    print(f"  global characters: {report['summary']['n_global_characters']}")
    print(f"  global arcs: {report['summary']['n_global_arcs']}")
    print(f"  schema violations: {report['summary']['n_schema_violations']}")
    print(f"  cost: ¥{report['cost_cny']}")
    print(f"\n  Field completeness:")
    for layer, fc in report['completeness'].items():
        avg = sum(fc.values()) / len(fc) if fc else 0
        print(f"    {layer:15s} avg={avg:.3f}  details={fc}")
    print(f"\n  Reference integrity:")
    ri = report['reference_integrity']
    print(f"    arc dangling refs: {len(ri['arc_dangling_event_refs'])}")
    print(f"    video main_arcs dangling: {len(ri['video_main_arcs_dangling'])}")
    print(f"    video key_events dangling: {len(ri['video_key_events_dangling'])}")
    print(f"    video char_refs matched: {ri['video_char_refs_matched_in_global']}/{ri['video_char_refs']}")
    print(f"\n  Character coverage:")
    cc = report['character_coverage']
    print(f"    event participants covered by global chars: {cc['covered_by_global_chars']}/{cc['unique_participants_in_events']}")
    if cc['missing']:
        print(f"    missing: {cc['missing']}")
    print(f"\n  Arc distribution:")
    for k, v in report['arc_distribution'].items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
