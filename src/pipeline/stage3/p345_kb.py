"""Stage 3 P3 (PlotArc) + P4 (Video 摘要) + P5 (Global 角色/Arc).

从原 scripts/stage3_full.py 抽出核心逻辑, 供 src.pipeline.stage3 编排。

输入:
  - data/output/{video}/stage3_dryrun.json (P1+P2 产物, 含 events)
  - data/output/_global/characters.json (跨集累积, 首次为空)
  - data/output/_global/global_arcs.json (跨集累积, 首次为空)
  - data/output/_global/video_summaries.json (跨集累积, 首次为空)

输出:
  - data/output/{video}/arc_updates.json (本集 P3 产出)
  - data/output/{video}/video_summary.json (本集 P4 产出)
  - data/output/{video}/stage3_kb.json (本集完整 KB 汇总)
  - 更新 data/output/_global/*.json (P5 跨集累积)
"""

from __future__ import annotations

import json
import os

from src.core.config import get_config
from src.core.cost import get_cost_tracker, reset_cost_tracker
from src.core.helpers.json_utils import load_json, save_json
from src.core.llm.qwen_text import QwenTextClient
from src.core.logging import get_logger
from src.pipeline.stage3.llm_json import call_llm_json

logger = get_logger()


# ============================================================
# P3: PlotArc
# ============================================================

def run_p3(client: QwenTextClient, video_id: str, events: list[dict],
           global_arcs: list[dict], prompt_template: str) -> list[dict]:
    """P3: 把本集 events 匹配到 global_arcs, 产出 arc_updates."""
    events_brief = json.dumps([
        {
            "event_id": e.get("event_id", ""),
            "title": e.get("title", ""),
            "participants": e.get("participants", []),
            "motivation": e.get("motivation", ""),
            "outcome": e.get("outcome", ""),
            "summary": e.get("summary", ""),
            "keywords": e.get("keywords", []),
        } for e in events
    ], ensure_ascii=False, indent=2)
    arcs_brief = json.dumps([
        {
            "arc_id": a.get("arc_id", ""),
            "title": a.get("title", ""),
            "type": a.get("type", ""),
            "participants": a.get("participants", []),
            "status": a.get("status", ""),
            "summary": a.get("summary", ""),
        } for a in global_arcs
    ], ensure_ascii=False, indent=2)

    prompt = (prompt_template
              .replace("{video_id}", video_id)
              .replace("{events_json}", events_brief)
              .replace("{global_arcs_json}", arcs_brief))

    logger.info(f"[P3] PlotArc 匹配 ({len(events)} events, {len(global_arcs)} 已有 arcs)...")
    result = call_llm_json(client, prompt, stage="stage3_p3_plot_arc",
                          max_tokens=6000, expected_key="arc_updates")
    if not isinstance(result, list):
        result = result.get("arc_updates", []) if isinstance(result, dict) else []
    logger.info(f"[P3] 产出 {len(result)} arc_updates")
    return result


# ============================================================
# P4: Video 单集摘要
# ============================================================

def run_p4(client: QwenTextClient, video_id: str, events: list[dict],
           arc_updates: list[dict], prompt_template: str) -> dict:
    """P4: 整集摘要 + character_refs + main_arcs."""
    events_brief = json.dumps([
        {
            "event_id": e.get("event_id", ""),
            "title": e.get("title", ""),
            "participants": e.get("participants", []),
            "summary": e.get("summary", ""),
        } for e in events
    ], ensure_ascii=False, indent=2)
    arcs_brief = json.dumps([
        {
            "arc_id": a.get("arc_id", ""),
            "title": a.get("title", ""),
            "type": a.get("type", ""),
            "status": a.get("status", ""),
        } for a in arc_updates
    ], ensure_ascii=False, indent=2)

    prompt = (prompt_template
              .replace("{video_id}", video_id)
              .replace("{events_json}", events_brief)
              .replace("{arc_updates_json}", arcs_brief))

    logger.info(f"[P4] Video 摘要生成...")
    result = call_llm_json(client, prompt, stage="stage3_p4_video_summary", max_tokens=3000)
    if not isinstance(result, dict):
        result = {"video_id": video_id, "title": "", "episode_summary": ""}
    logger.info(f"[P4] title={result.get('title', '')!r}, "
                f"n_chars={len(result.get('character_refs', []))}, "
                f"n_arcs={len(result.get('main_arcs', []))}")
    return result


# ============================================================
# P5: Global characters + arcs 终态
# ============================================================

def run_p5(client: QwenTextClient, existing_characters: list[dict],
           video_summaries: list[dict], all_arc_updates: list[dict],
           prompt_template: str) -> dict:
    """P5: 跨集角色归一 + arc 终态汇总."""
    prompt = (prompt_template
              .replace("{existing_characters_json}",
                       json.dumps(existing_characters, ensure_ascii=False, indent=2))
              .replace("{video_summaries_json}",
                       json.dumps(video_summaries, ensure_ascii=False, indent=2))
              .replace("{all_arc_updates_json}",
                       json.dumps(all_arc_updates, ensure_ascii=False, indent=2)))

    logger.info(f"[P5] Global characters + arcs 更新 (输入 {len(video_summaries)} videos, "
                f"{len(existing_characters)} 已有角色)...")
    result = call_llm_json(client, prompt, stage="stage3_p5_global", max_tokens=8000)
    if not isinstance(result, dict):
        result = {"characters": [], "global_arcs": []}
    logger.info(f"[P5] characters={len(result.get('characters', []))}, "
                f"global_arcs={len(result.get('global_arcs', []))}")
    return result


# ============================================================
# 全局状态 (跨集累积)
# ============================================================

GLOBAL_DIR_NAME = "_global"


def _global_dir() -> str:
    cfg = get_config()
    g = os.path.join(cfg.output_root, GLOBAL_DIR_NAME)
    os.makedirs(g, exist_ok=True)
    return g


def _load_global_state() -> tuple[list[dict], list[dict], list[dict]]:
    """加载跨集累积状态: (characters, global_arcs, video_summaries)."""
    g = _global_dir()
    chars_p = os.path.join(g, "characters.json")
    arcs_p = os.path.join(g, "global_arcs.json")
    vids_p = os.path.join(g, "video_summaries.json")
    chars = load_json(chars_p) if os.path.isfile(chars_p) else []
    arcs = load_json(arcs_p) if os.path.isfile(arcs_p) else []
    vids = load_json(vids_p) if os.path.isfile(vids_p) else []
    return chars, arcs, vids


def _save_global_state(chars: list, arcs: list, vids: list):
    g = _global_dir()
    save_json(chars, os.path.join(g, "characters.json"))
    save_json(arcs, os.path.join(g, "global_arcs.json"))
    save_json(vids, os.path.join(g, "video_summaries.json"))


# ============================================================
# 主流程
# ============================================================

def run_p345(video_dir: str, enable_thinking: bool = False) -> dict:
    """跑 P3 + P4 + P5, 消费 stage3_dryrun.json, 返回完整 KB dict.

    Args:
        video_dir: 视频目录名 (相对 output_root)
        enable_thinking: 是否开启 thinking (慢但更准)

    Returns:
        完整 KB dict (events / actions / arc_updates / video_summary / cost)
    """
    cfg = get_config()
    output_dir = os.path.join(cfg.output_root, video_dir)
    dryrun_path = os.path.join(output_dir, "stage3_dryrun.json")
    if not os.path.isfile(dryrun_path):
        raise FileNotFoundError(
            f"未找到 {dryrun_path}; 先跑 P1+P2 (run_p1p2 或 scripts/stage3_p1p2)"
        )

    reset_cost_tracker()
    dryrun = load_json(dryrun_path)
    events = dryrun.get("events", [])
    if not events:
        raise ValueError("stage3_dryrun.json 中没有 events")

    video_id = dryrun.get("video_id", video_dir)
    logger.info(f"{'='*60}\nStage 3 P3-P5: {video_id}\n  events: {len(events)}\n  "
                f"thinking: {enable_thinking}\n{'='*60}")

    # 加载 prompt
    prompts = cfg.prompts
    p3_prompt = prompts["stage3_p3_plot_arc"]["user"]
    p4_prompt = prompts["stage3_p4_video_summary"]["user"]
    p5_prompt = prompts["stage3_p5_global"]["user"]

    client = QwenTextClient()

    # 加载全局状态
    existing_chars, existing_arcs, existing_vids = _load_global_state()
    logger.info(f"[Init] global state: {len(existing_chars)} chars, "
                f"{len(existing_arcs)} arcs, {len(existing_vids)} videos")

    # P3: PlotArc
    arc_updates = run_p3(client, video_id, events, existing_arcs, p3_prompt)

    # P4: Video Summary
    video_summary = run_p4(client, video_id, events, arc_updates, p4_prompt)

    # P5: Global (用累积的 video_summaries + 当前 arc_updates + 已有 characters)
    updated_vids = list(existing_vids)
    # 替换或追加本集
    updated_vids = [v for v in updated_vids if v.get("video_id") != video_id]
    updated_vids.append(video_summary)

    # 累积所有 arc_updates (按集分组)
    all_arc_updates = list(existing_arcs)  # 简化: 用现有 arcs 作 base
    # 把本集 arc_updates 加进去 (新 arc 用 arc_new_xxx, 旧 arc 保留 id)
    all_arc_updates.extend([a for a in arc_updates if a.get("update_type") == "new"])

    p5_result = run_p5(client, existing_chars, updated_vids, all_arc_updates, p5_prompt)
    new_chars = p5_result.get("characters", [])
    new_arcs = p5_result.get("global_arcs", [])

    # 保存本集产物
    save_json(arc_updates, os.path.join(output_dir, "arc_updates.json"))
    save_json(video_summary, os.path.join(output_dir, "video_summary.json"))

    # 本集完整 KB
    kb = {
        "video_id": video_id,
        "events": events,
        "actions": dryrun.get("actions", []),
        "arc_updates": arc_updates,
        "video_summary": video_summary,
        "cost": get_cost_tracker().report_dict(),
    }
    save_json(kb, os.path.join(output_dir, "stage3_kb.json"))

    # 更新全局
    _save_global_state(new_chars, new_arcs, updated_vids)

    # 打印 cost
    logger.info(f"{'='*60}\nCost Report\n{'='*60}")
    logger.info(get_cost_tracker().report())

    return kb
