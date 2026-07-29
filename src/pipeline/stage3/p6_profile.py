"""Stage 3 P6: 角色深度画像.

消费 characters.json (P5) + 所有集的 events + actions (P1/P2 产出),
单次 LLM 调用产出 per-character 的性格深层 + 行为模式带例证.

输出: data/output/_global/character_profiles.json

设计决策 (见 plan):
  - all-in-one 单次调用 (不 per-character), 因为 events.participants 归一不可靠
    (同一角色多种 char_id), 让 LLM 用 name+aliases 全局匹配
  - actions 从 stage3_dryrun.json 顶层取 (含 utterance), 不用 events.actions
    (后者只有 action_id+target, 无原话)
  - 输入字段精简: events 去 actions 数组, actions 去 evidence (控输入大小)
"""

from __future__ import annotations

import glob
import json
import os

from src.core.config import get_config
from src.core.cost import get_cost_tracker, reset_cost_tracker
from src.core.helpers.json_utils import load_json, save_json
from src.core.llm.qwen_text import QwenTextClient
from src.core.logging import get_logger
from src.pipeline.stage3.llm_json import call_llm_json
from src.pipeline.stage3.p345_kb import _global_dir

logger = get_logger()


def _aggregate_all_episodes() -> tuple[list[dict], list[dict], list[str]]:
    """扫描所有 data/output/*/stage3_dryrun.json, 聚合 events + actions (精简字段).

    Returns:
        (events, actions, episode_ids) — episode_ids 为已建库的集清单
    """
    cfg = get_config()
    output_root = cfg.output_root
    events: list[dict] = []
    actions: list[dict] = []
    episode_ids: list[str] = []

    if not os.path.isdir(output_root):
        return events, actions, episode_ids

    # 递归查找所有 stage3_dryrun.json (支持 data/output/{作品}/{季}/{集}/ 多级嵌套)
    pattern = os.path.join(output_root, "**", "stage3_dryrun.json")
    for dryrun_path in sorted(glob.glob(pattern, recursive=True)):
        ep_dir = os.path.dirname(dryrun_path)
        episode_id = os.path.relpath(ep_dir, output_root).replace("\\", "/")
        if episode_id.startswith("_"):
            continue  # 跳过 _global / _batch_reports

        data = load_json(dryrun_path)
        episode_ids.append(episode_id)

        # events 精简: 去掉 actions 数组 (顶层 actions 单独喂), 保留语义字段
        for e in data.get("events", []):
            events.append({
                "event_id": e.get("event_id", ""),
                "video_id": e.get("video_id", episode_id),
                "title": e.get("title", ""),
                "participants": e.get("participants", []),
                "motivation": e.get("motivation", ""),
                "outcome": e.get("outcome", ""),
                "summary": e.get("summary", ""),
            })

        # actions 精简: 去掉 evidence (太大), 保留 utterance 用于行为分析
        for a in data.get("actions", []):
            actions.append({
                "action_id": a.get("action_id", ""),
                "actor": a.get("actor", ""),
                "action": a.get("action", ""),
                "emotion": a.get("emotion", ""),
                "utterance": a.get("utterance", ""),
            })

    return events, actions, episode_ids


def run_p6(enable_thinking: bool = False) -> dict:
    """跑 P6 角色深度画像, 写 _global/character_profiles.json.

    Args:
        enable_thinking: 开启 qwen thinking (调 prompt 阶段用, token 涨 3-4 倍)

    Returns:
        {meta, cost, profiles} — meta 含聚合统计, profiles 为画像列表

    Raises:
        FileNotFoundError: 没有 characters.json (需先跑 P5)
        ValueError: 没有任何 stage3_dryrun.json
    """
    reset_cost_tracker()
    cfg = get_config()
    gdir = _global_dir()

    # 1. 加载 characters.json (P5 产出)
    chars_path = os.path.join(gdir, "characters.json")
    if not os.path.isfile(chars_path):
        raise FileNotFoundError(
            f"未找到 {chars_path}; 先跑 P5 (run_p345 或 scripts.stage3_p345) 建立角色清单"
        )
    characters = load_json(chars_path)
    logger.info(f"[P6] 加载 characters.json: {len(characters)} 个角色")

    # 2. 聚合所有集的 events + actions
    events, actions, episode_ids = _aggregate_all_episodes()
    if not events:
        raise ValueError(
            "未找到任何 stage3_dryrun.json; 先跑 P1+P2 (run_p1p2 或 scripts.stage3_p1p2)"
        )
    logger.info(f"[P6] 聚合 {len(episode_ids)} 集: {len(events)} events, {len(actions)} actions")
    logger.info(f"[P6] 涉及集: {episode_ids}")

    # 3. characters 精简 (只留 P6 匹配需要的字段, 去掉 relationships 等大字段)
    chars_brief = [
        {
            "character_id": c.get("character_id", ""),
            "name": c.get("name", ""),
            "aliases": c.get("aliases", []),
            "role": c.get("role", ""),
        }
        for c in characters
    ]

    # 4. 构建 prompt (替换占位符)
    prompt_template = cfg.prompts["stage3_p6_character_profile"]["user"]
    prompt = (
        prompt_template
        .replace("__INPUT_CHARACTERS_JSON__", json.dumps(chars_brief, ensure_ascii=False, indent=2))
        .replace("__INPUT_EVENTS_JSON__", json.dumps(events, ensure_ascii=False, indent=2))
        .replace("__INPUT_ACTIONS_JSON__", json.dumps(actions, ensure_ascii=False, indent=2))
    )

    # 5. 单次 LLM 调用 (all-in-one)
    client = QwenTextClient()
    logger.info(f"[P6] 调用 LLM (all-in-one, thinking={enable_thinking})...")
    result = call_llm_json(
        client, prompt, stage="stage3_p6_profile",
        max_tokens=8000, expected_key="profiles",
        enable_thinking=enable_thinking,
    )
    profiles = result if isinstance(result, list) else []
    logger.info(f"[P6] LLM 产出 {len(profiles)} 个角色画像")

    # 6. 保存
    output = {
        "meta": {
            "n_characters_source": len(characters),
            "n_profiles": len(profiles),
            "n_episodes": len(episode_ids),
            "episode_ids": episode_ids,
            "n_events": len(events),
            "n_actions": len(actions),
        },
        "cost": get_cost_tracker().report_dict(),
        "profiles": profiles,
    }
    out_path = os.path.join(gdir, "character_profiles.json")
    save_json(output, out_path)
    logger.info(f"[P6] 保存 → {out_path}")

    # Cost 报告
    logger.info("=" * 70)
    logger.info("Cost 报告")
    logger.info("=" * 70)
    logger.info(get_cost_tracker().report())

    return output
