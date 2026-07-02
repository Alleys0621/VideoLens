"""Stage 3 P6 角色画像质量评估.

评估 data/output/_global/character_profiles.json:
  - personality 字段完整率 (6 维: core_motivation/values/strengths/weaknesses/inner_conflict/growth_direction)
  - behavior_patterns 数量分布 + 孤立 pattern 检测 (evidence < 2 条)
  - evidence 引用有效性 (event_id / action_id 在聚合数据里是否存在, dangling ref 检查)
  - 覆盖率 (有画像的角色 / characters.json 总数)

输出: data/output/_global/profile_quality_report.json

用法: python -m src.eval.stage3_profile_quality
"""

from __future__ import annotations

import os
from collections import Counter

from src.core.config import get_config
from src.core.helpers.json_utils import load_json, save_json
from src.core.logging import get_logger
from src.eval._quality_utils import _field_completeness
from src.pipeline.stage3.p6_profile import _aggregate_all_episodes

logger = get_logger()

PERSONALITY_FIELDS = [
    "core_motivation", "values", "strengths",
    "weaknesses", "inner_conflict", "growth_direction",
]


def evaluate(
    profiles_data: dict,
    all_event_ids: set,
    all_action_ids: set,
    characters: list,
) -> dict:
    """评估 character_profiles, 返回报告 dict.

    Args:
        profiles_data: character_profiles.json 完整内容 (含 meta + profiles)
        all_event_ids: 所有集聚合的 event_id 集合 (dangling 检查基准)
        all_action_ids: 所有集聚合的 action_id 集合
        characters: characters.json 角色列表 (覆盖率基准)
    """
    profiles = profiles_data.get("profiles", []) or []

    # ---------- personality 字段完整率 ----------
    personality_records = [p.get("personality", {}) or {} for p in profiles]
    personality_completeness = _field_completeness(personality_records, PERSONALITY_FIELDS)

    # ---------- behavior_patterns 统计 + 孤立 pattern ----------
    n_patterns_total = 0
    n_evidence_total = 0
    lonely_patterns = []  # evidence < 2 (违反 prompt 硬约束)
    patterns_per_profile = []

    for p in profiles:
        patterns = p.get("behavior_patterns", []) or []
        patterns_per_profile.append(len(patterns))
        n_patterns_total += len(patterns)
        for pat in patterns:
            ev = pat.get("evidence", []) or []
            n_evidence_total += len(ev)
            if len(ev) < 2:
                lonely_patterns.append({
                    "character_id": p.get("character_id", "?"),
                    "pattern": pat.get("pattern", "")[:60],
                    "n_evidence": len(ev),
                })

    avg_patterns = round(n_patterns_total / len(profiles), 2) if profiles else 0
    avg_evidence = round(n_evidence_total / n_patterns_total, 2) if n_patterns_total else 0

    # ---------- evidence dangling ref 检查 ----------
    dangling_event_refs = []
    dangling_action_refs = []
    for p in profiles:
        for pat in p.get("behavior_patterns", []) or []:
            for ev in pat.get("evidence", []) or []:
                eid = ev.get("event_id", "")
                aid = ev.get("action_id", "")
                if eid and eid not in all_event_ids:
                    dangling_event_refs.append({
                        "character_id": p.get("character_id"),
                        "event_id": eid,
                    })
                if aid and aid not in all_action_ids:
                    dangling_action_refs.append({
                        "character_id": p.get("character_id"),
                        "action_id": aid,
                    })

    # ---------- 覆盖率 ----------
    source_char_ids = {c.get("character_id") for c in characters if c.get("character_id")}
    profile_char_ids = {p.get("character_id") for p in profiles if p.get("character_id")}
    covered = profile_char_ids & source_char_ids
    missing = source_char_ids - profile_char_ids

    return {
        "summary": {
            "n_profiles": len(profiles),
            "n_characters_source": len(source_char_ids),
            "n_covered": len(covered),
            "coverage_ratio": round(len(covered) / len(source_char_ids), 4) if source_char_ids else 0.0,
            "n_patterns": n_patterns_total,
            "n_dangling_event_refs": len(dangling_event_refs),
            "n_dangling_action_refs": len(dangling_action_refs),
            "n_lonely_patterns": len(lonely_patterns),
        },
        "personality_completeness": personality_completeness,
        "behavior_patterns": {
            "avg_per_profile": avg_patterns,
            "avg_evidence_per_pattern": avg_evidence,
            "distribution": dict(Counter(patterns_per_profile)),
        },
        "lonely_patterns": lonely_patterns,
        "dangling_refs": {
            "events": dangling_event_refs[:20],   # 截断避免报告过大
            "actions": dangling_action_refs[:20],
        },
        "missing_character_ids": sorted(missing),
    }


def main() -> dict:
    """跑评估, 写报告, 返回报告 dict."""
    cfg = get_config()
    from src.pipeline.stage3.p345_kb import _global_dir
    gdir = _global_dir()

    profiles_path = os.path.join(gdir, "character_profiles.json")
    if not os.path.isfile(profiles_path):
        raise FileNotFoundError(f"未找到 {profiles_path}; 先跑 P6 (python -m scripts.stage3_p6)")

    chars_path = os.path.join(gdir, "characters.json")
    characters = load_json(chars_path) if os.path.isfile(chars_path) else []
    profiles_data = load_json(profiles_path)

    # 聚合所有 event_id / action_id (dangling 检查基准)
    events, actions, _ = _aggregate_all_episodes()
    all_event_ids = {e["event_id"] for e in events if e.get("event_id")}
    all_action_ids = {a["action_id"] for a in actions if a.get("action_id")}
    logger.info(f"评估基准: {len(all_event_ids)} events, {len(all_action_ids)} actions")

    report = evaluate(profiles_data, all_event_ids, all_action_ids, characters)

    out_path = os.path.join(gdir, "profile_quality_report.json")
    save_json(report, out_path)
    logger.info(f"画像质量报告: {out_path}")

    # 打印关键指标
    print("\n" + "=" * 70)
    print("P6 角色画像质量")
    print("=" * 70)
    s = report["summary"]
    print(f"\n[1] 覆盖")
    print(f"  profiles: {s['n_profiles']} / source {s['n_characters_source']} "
          f"(coverage {s['coverage_ratio']:.2%})")
    if report["missing_character_ids"]:
        print(f"  missing: {report['missing_character_ids']}")

    print(f"\n[2] personality 字段完整率")
    for k, v in report["personality_completeness"].items():
        print(f"  {k}: {v}")

    bp = report["behavior_patterns"]
    print(f"\n[3] behavior_patterns")
    print(f"  总 {s['n_patterns']} 个 (avg {bp['avg_per_profile']}/profile, "
          f"avg {bp['avg_evidence_per_pattern']} evidence/pattern)")
    print(f"  分布 (patterns→profile数): {bp['distribution']}")

    print(f"\n[4] 质量问题")
    print(f"  孤立 pattern (evidence<2): {s['n_lonely_patterns']}")
    print(f"  dangling event refs: {s['n_dangling_event_refs']}")
    print(f"  dangling action refs: {s['n_dangling_action_refs']}")

    print("\n" + "=" * 70)

    return report


if __name__ == "__main__":
    main()
