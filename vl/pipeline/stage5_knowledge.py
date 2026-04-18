"""Stage 5: 结构化知识库生成

基于 Stage 4 的时序对齐数据，增量生成三层结构化知识库。

三层结构:
  Layer 1: 视频名称
  Layer 2: 大事件 (叙事事件，基于场景变化)
  Layer 3: 小事件 (场景内的具体动作/互动)

输出: knowledge_base.json
"""

import json
import os

from vl.core.config import AppConfig
from vl.core.helpers.json_utils import save_json

from vl.core.logging import get_logger

logger = get_logger()


def run_stage5(
    video_id: str,
    video_title: str,
    genre: str,
    aligned_timeline: dict,
    output_dir: str,
    config: AppConfig,
) -> dict:
    """
    执行 Stage 5: 生成结构化知识库。

    Args:
        aligned_timeline: Stage 4 输出的时序对齐数据

    Returns:
        knowledge_base: 三层结构化知识库 dict
    """
    logger.info("[Stage 5] 开始生成结构化知识库...")

    events = aligned_timeline.get("events", [])
    scenes = aligned_timeline.get("scenes", [])

    if not events and not scenes:
        logger.warning("无可用的对齐数据，跳过知识库生成")
        return {}

    knowledge_base = {}

    if config.dashscope_api_key:
        # 使用 LLM 增量生成知识库
        knowledge_base = _generate_knowledge_base_with_llm(
            video_title=video_title,
            genre=genre,
            events=events,
            scenes=scenes,
            config=config,
        )
    else:
        # 无 API Key → 基于规则直接构建
        logger.info("未配置 API Key，使用规则模式构建知识库")
        knowledge_base = _build_knowledge_base_by_rules(
            video_title=video_title,
            events=events,
            scenes=scenes,
        )

    # 保存知识库
    kb_dir = os.path.join(output_dir, "knowledge", video_id)
    os.makedirs(kb_dir, exist_ok=True)
    kb_path = os.path.join(kb_dir, "knowledge_base.json")
    save_json(knowledge_base, kb_path)

    # 统计
    event_count = len(knowledge_base.get(video_title, {}))
    sub_event_count = sum(
        len(subs) for subs in knowledge_base.get(video_title, {}).values()
    )
    logger.info(f"[Stage 5] 知识库生成完成: {event_count} 个大事件, {sub_event_count} 个小事件")
    logger.info(f"知识库已保存: {kb_path}")

    return knowledge_base


def _generate_knowledge_base_with_llm(
    video_title: str,
    genre: str,
    events: list[dict],
    scenes: list[dict],
    config: AppConfig,
) -> dict:
    """使用 LLM 增量生成结构化知识库"""
    from vl.core.llm.qwen_text import QwenTextClient
    from vl.core.llm.base_client import BaseLLMClient

    text_client = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
    json_extractor = BaseLLMClient(model=config.model_text, api_key=config.dashscope_api_key)

    prompts = config.prompts.get("knowledge_summary", {})
    user_template = prompts.get("user", "")

    # 初始知识库为空
    summary = "{}"

    # 按叙事事件逐个增量更新
    for i, event in enumerate(events):
        # 跳过片头曲/片尾曲事件
        if event.get("content_type") in ("opening", "ending"):
            ct_label = "片头曲" if event["content_type"] == "opening" else "片尾曲"
            logger.info("跳过 %s 事件 %d/%d", ct_label, i + 1, len(events))
            continue

        logger.info("知识库增量更新: 事件 %d/%d", i + 1, len(events))

        # 构建当前事件的 caption 信息
        caption = _build_event_caption(event, scenes)

        # 拼接当前事件的台词
        audiotext = event.get("transcript", "") or "(无台词)"

        if user_template:
            prompt = user_template.format(
                video_title=video_title,
                genre=genre,
                caption=caption,
                audiotext=audiotext,
                summary=summary,
            )
        else:
            prompt = (
                f"基于以下信息更新知识库。\n"
                f"场景描述: {caption}\n"
                f"台词: {audiotext}\n"
                f"已有知识库: {summary}\n"
                f"输出严格的 JSON。"
            )

        raw = text_client.generate(prompt, system=prompts.get("system", ""))
        if raw:
            extracted = json_extractor.extract_json(raw)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    if isinstance(parsed, dict):
                        summary = json.dumps(parsed, ensure_ascii=False)
                except json.JSONDecodeError:
                    logger.warning(f"事件 {i+1} JSON 解析失败，保留上一次知识库")

    # 解析最终结果
    try:
        return json.loads(summary)
    except json.JSONDecodeError:
        logger.warning("最终知识库 JSON 解析失败，回退到规则模式")
        return _build_knowledge_base_by_rules(video_title, events, scenes)


def _build_knowledge_base_by_rules(
    video_title: str,
    events: list[dict],
    scenes: list[dict],
) -> dict:
    """
    基于规则构建知识库 (无 LLM 时的回退方案)。

    规则:
    - 每个 narrative event → 一个大事件
    - 大事件内的每个场景 → 一个小事件
    - 小事件名 = main_actions 或 "场景N"
    - 小事件描述 = transcript 摘要
    """
    kb = {}

    for i, event in enumerate(events):
        # 跳过片头曲/片尾曲事件
        if event.get("content_type") in ("opening", "ending"):
            continue

        # 大事件命名: "第N幕: 白天/晚上 + 室内/室外"
        tod = event.get("time_of_day", "未知")
        scene_type = event.get("scene_type", "未知")
        space = event.get("space", "")

        event_name = f"第{i+1}幕: {tod}{scene_type}"
        if space and space != "未知":
            event_name += f"--{space}"

        # 小事件: 从事件内的场景中提取
        sub_events = {}
        scene_ids = event.get("scene_ids", [])

        for j, scene_id in enumerate(scene_ids):
            scene = next((s for s in scenes if s["scene_id"] == scene_id), None)
            if not scene:
                continue

            visual = scene.get("visual", {})
            action = visual.get("main_actions", "")
            transcript = scene.get("transcript", "")

            if action:
                sub_name = action[:20]
            elif transcript:
                sub_name = transcript[:20]
            else:
                sub_name = f"场景{scene.get('index', j)}"

            # 描述
            parts = []
            if action:
                parts.append(action)
            interactions = visual.get("interactions", "")
            if interactions:
                parts.append(interactions)
            desc = "；".join(parts) if parts else (transcript[:20] if transcript else "无内容")

            sub_events[sub_name] = desc

        if sub_events:
            kb[event_name] = sub_events

    return {video_title: kb}


def _build_event_caption(event: dict, scenes: list[dict] = None) -> str:
    """将叙事事件转为 caption JSON 字符串供 LLM 使用"""
    caption = {
        "time_of_day": event.get("time_of_day", "未知"),
        "scene_type": event.get("scene_type", "未知"),
        "space": event.get("space", "未知"),
        "characters": event.get("characters", []),
        "scene_count": event.get("scene_count", 0),
        "duration": event.get("duration", 0),
    }

    # 提取事件内各场景的动作
    if scenes:
        scene_map = {s["scene_id"]: s for s in scenes}
        actions = []
        interactions = []
        for scene_id in event.get("scene_ids", []):
            scene = scene_map.get(scene_id)
            if not scene:
                continue
            visual = scene.get("visual", {})
            if visual.get("main_actions"):
                actions.append(visual["main_actions"])
            if visual.get("interactions"):
                interactions.append(visual["interactions"])
        if actions:
            caption["main_actions"] = "；".join(actions)
        if interactions:
            caption["interactions"] = "；".join(interactions)

    return json.dumps(caption, ensure_ascii=False)
