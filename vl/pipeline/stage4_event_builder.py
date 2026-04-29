"""Stage 4: Event Builder - 从连续帧描述中提取事件

从 Stage 3 的原子化帧描述中，使用 LLM 提取结构化事件。

流程:
  1. 按 scene/time_of_day 连续性分组
  2. 每组收集所有帧的 actions
  3. LLM 提取事件 (合并连续动作为事件)
  4. 保存 events.json
"""

import json
import os

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.helpers.json_utils import save_json

from vl.core.logging import get_logger

logger = get_logger()


def run_stage4(
    video_id: str,
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    output_dir: str,
    config: AppConfig,
) -> list[dict]:
    """
    执行 Stage 4: Event Builder。

    Args:
        scenes: 含 structured_caption 的场景列表 (Stage 3 输出)
        scene_transcripts: {scene_id: [text, ...]}

    Returns:
        events: 提取的事件列表
    """
    logger.info("[Stage 4] Event Builder: 从帧描述中提取事件...")

    # 1. 筛选有描述的场景
    captioned_scenes = [s for s in scenes if s.structured_caption]
    if not captioned_scenes:
        logger.warning("没有带描述的场景，跳过 Event Builder")
        return []

    # 2. 按连续性分组
    groups = _group_by_continuity(captioned_scenes)
    logger.info(f"分为 {len(groups)} 个连续组")

    # 3. 逐组提取事件
    all_events = []
    if config.dashscope_api_key:
        all_events = _extract_events_with_llm(groups, scene_transcripts, config)
    else:
        logger.info("未配置 API Key，使用规则模式")
        all_events = _extract_events_by_rules(groups)

    # 4. 保存
    events_dir = os.path.join(output_dir, "stage4_events", video_id)
    os.makedirs(events_dir, exist_ok=True)
    save_json(all_events, os.path.join(events_dir, "events.json"))

    logger.info(f"[Stage 4] 提取完成: {len(all_events)} 个事件")
    return all_events


# ──────────────────────────────────────────────────────────
# 分组: 按连续性 (同一 scene + time_of_day)
# ──────────────────────────────────────────────────────────

def _group_by_continuity(scenes: list[Scene]) -> list[list[Scene]]:
    """将场景按 scene/time_of_day 连续性分组"""
    if not scenes:
        return []

    groups = []
    current = [scenes[0]]
    prev_attrs = _get_attrs(scenes[0])

    for scene in scenes[1:]:
        attrs = _get_attrs(scene)
        if _is_continuous(prev_attrs, attrs):
            current.append(scene)
        else:
            groups.append(current)
            current = [scene]
        prev_attrs = attrs

    if current:
        groups.append(current)

    return groups


def _get_attrs(scene: Scene) -> dict:
    cap = scene.structured_caption or {}
    return {
        "scene": cap.get("scene", "未知"),
        "time_of_day": cap.get("time_of_day", "未知"),
    }


def _is_continuous(prev: dict, curr: dict) -> bool:
    """scene 或 time_of_day 变化 → 断组"""
    if prev["scene"] != "未知" and curr["scene"] != "未知" and prev["scene"] != curr["scene"]:
        return False
    if prev["time_of_day"] != "未知" and curr["time_of_day"] != "未知" and prev["time_of_day"] != curr["time_of_day"]:
        return False
    return True


# ──────────────────────────────────────────────────────────
# LLM 事件提取
# ──────────────────────────────────────────────────────────

def _extract_events_with_llm(
    groups: list[list[Scene]],
    scene_transcripts: dict[str, list[str]],
    config: AppConfig,
) -> list[dict]:
    """使用 LLM 逐组提取事件"""
    from vl.core.llm.qwen_text import QwenTextClient
    from vl.core.llm.base_client import BaseLLMClient

    text_client = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
    json_extractor = BaseLLMClient(model=config.model_text, api_key=config.dashscope_api_key)

    prompts = config.prompts.get("event_builder", {})
    user_template = prompts.get("user", "")
    system_prompt = prompts.get("system", "")

    if not user_template:
        logger.warning("event_builder prompt 未配置，跳过 LLM 事件提取")
        return []

    all_events = []

    for i, group in enumerate(groups):
        # 构建帧描述序列
        frame_descs = _build_frame_descriptions(group, scene_transcripts)
        if not frame_descs.strip():
            continue

        logger.info("Event Builder: 组 %d/%d (%d 帧)", i + 1, len(groups), len(group))

        prompt = user_template.format(frame_descriptions=frame_descs)

        raw = text_client.generate(prompt, system=system_prompt)
        if not raw:
            continue

        extracted = json_extractor.extract_json(raw)
        if not extracted:
            continue

        events = _parse_events(extracted, len(all_events), group)
        all_events.extend(events)

    return all_events


def _build_frame_descriptions(
    group: list[Scene],
    scene_transcripts: dict[str, list[str]],
) -> str:
    """将一组场景转为帧描述文本"""
    lines = []
    for scene in group:
        cap = scene.structured_caption or {}
        # 兼容 main_actions(string) 和 actions(list)
        actions = cap.get("main_actions", "")
        if not actions:
            raw = cap.get("actions", [])
            actions = "；".join(str(a) for a in raw) if isinstance(raw, list) else str(raw)

        characters = cap.get("characters", [])
        interaction = cap.get("interactions", "") or cap.get("interaction", "")
        transcript = " ".join(scene_transcripts.get(scene.scene_id, []))

        parts = [f"帧{scene.index} ({scene.start_time:.1f}s-{scene.end_time:.1f}s)"]
        if characters:
            parts.append(f"角色: {', '.join(str(c) for c in characters)}")
        if actions:
            parts.append(f"动作: {actions}")
        if interaction:
            parts.append(f"互动: {interaction}")
        if transcript:
            parts.append(f"台词: {transcript[:100]}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)


def _parse_events(raw_json: str, offset: int, group: list[Scene]) -> list[dict]:
    """解析 LLM 输出的事件 JSON"""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = [data]

    events = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        events.append({
            "event_id": f"E{offset + idx:03d}",
            "characters": item.get("characters", []),
            "goal": item.get("goal", "未知"),
            "actions": item.get("actions", []),
            "result": item.get("result", "未知"),
            "event_type": item.get("event_type", "未知"),
            "scene_ids": [s.scene_id for s in group],
            "start_time": group[0].start_time,
            "end_time": group[-1].end_time,
        })

    return events


# ──────────────────────────────────────────────────────────
# 规则模式 (无 LLM 时的回退)
# ──────────────────────────────────────────────────────────

def _extract_events_by_rules(groups: list[list[Scene]]) -> list[dict]:
    """基于规则提取事件 (每个组 = 一个事件)"""
    events = []
    for idx, group in enumerate(groups):
        # 收集所有动作
        actions = []
        characters = set()
        for scene in group:
            cap = scene.structured_caption or {}
            act = cap.get("main_actions", "")
            if not act:
                raw = cap.get("actions", [])
                act = "；".join(str(a) for a in raw) if isinstance(raw, list) else str(raw)
            if act:
                actions.append(act)
            for c in cap.get("characters", []):
                characters.add(str(c))

        events.append({
            "event_id": f"E{idx:03d}",
            "characters": list(characters),
            "goal": "未知",
            "actions": actions,
            "result": "未知",
            "event_type": "未知",
            "scene_ids": [s.scene_id for s in group],
            "start_time": group[0].start_time,
            "end_time": group[-1].end_time,
        })

    return events
