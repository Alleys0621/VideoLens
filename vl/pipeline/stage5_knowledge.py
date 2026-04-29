"""Stage 5: 结构化知识库生成

基于 enriched scenes，逐帧增量生成三层结构化知识库。

三层结构:
  Layer 1: 视频名称
  Layer 2: 大事件 (场景变化时创建，如室内→室外)
  Layer 3: 小事件 (帧内的具体人物/动作/互动)

输出: stage5_knowledge/{video_id}/knowledge_base.json
"""

import json
import os

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.helpers.json_utils import save_json

from vl.core.logging import get_logger

logger = get_logger()


def run_stage5(
    video_id: str,
    video_title: str,
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    output_dir: str,
    config: AppConfig,
) -> dict:
    """
    执行 Stage 5: 生成结构化知识库。

    Args:
        scenes: 含 structured_caption 的场景列表 (Stage 3 输出)
        scene_transcripts: {scene_id: [text, ...]}

    Returns:
        knowledge_base: 三层结构化知识库 dict
    """
    logger.info("[Stage 5] 开始生成结构化知识库...")

    if not scenes:
        logger.warning("无可用的场景数据，跳过知识库生成")
        return {}

    knowledge_base = {}

    if config.dashscope_api_key:
        knowledge_base = _generate_knowledge_base_with_llm(
            video_title=video_title,
            scenes=scenes,
            scene_transcripts=scene_transcripts,
            config=config,
        )
    else:
        logger.info("未配置 API Key，使用规则模式构建知识库")
        knowledge_base = _build_knowledge_base_by_rules(
            video_title=video_title,
            scenes=scenes,
        )

    # 保存知识库
    kb_dir = os.path.join(output_dir, "stage5_knowledge", video_id)
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
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    config: AppConfig,
) -> dict:
    """使用 LLM 逐帧增量生成结构化知识库"""
    from vl.core.llm.qwen_text import QwenTextClient
    from vl.core.llm.base_client import BaseLLMClient

    text_client = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
    json_extractor = BaseLLMClient(model=config.model_text, api_key=config.dashscope_api_key)

    prompts = config.prompts.get("knowledge_summary", {})
    user_template = prompts.get("user", "")
    system_prompt = prompts.get("system", "")

    if not user_template:
        logger.warning("knowledge_summary prompt 未配置，回退到规则模式")
        return _build_knowledge_base_by_rules(video_title, scenes)

    # 初始知识库为空
    summary = "{}"

    total = len(scenes)
    for i, scene in enumerate(scenes):
        # 跳过片头曲/片尾曲
        content_type = scene.content_type or "main"
        if content_type in ("opening", "ending"):
            ct_label = "片头曲" if content_type == "opening" else "片尾曲"
            logger.info("跳过 %s 场景 %d/%d", ct_label, i + 1, total)
            continue

        logger.info("知识库增量更新: 帧 %d/%d", i + 1, total)

        # 构建当前帧的 caption JSON
        caption = _build_scene_caption(scene)

        # 当前帧的台词
        audiotext = " ".join(scene_transcripts.get(scene.scene_id, [])) or "(无台词)"

        prompt = user_template.format(
            video_title=video_title,
            caption=caption,
            audiotext=audiotext,
            summary=summary,
        )

        raw = text_client.generate(prompt, system=system_prompt)
        if raw:
            extracted = json_extractor.extract_json(raw)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    if isinstance(parsed, dict):
                        summary = json.dumps(parsed, ensure_ascii=False)
                except json.JSONDecodeError:
                    logger.warning(f"帧 {i+1} JSON 解析失败，保留上一次知识库")

    # 解析最终结果
    try:
        return json.loads(summary)
    except json.JSONDecodeError:
        logger.warning("最终知识库 JSON 解析失败，回退到规则模式")
        return _build_knowledge_base_by_rules(video_title, scenes)


def _build_scene_caption(scene: Scene) -> str:
    """将单个场景的结构化描述转为 caption JSON 字符串"""
    cap = scene.structured_caption or {}
    if not cap:
        return json.dumps({"scene_id": scene.scene_id}, ensure_ascii=False)

    # 兼容 actions(list) 和 main_actions(string)
    actions_val = cap.get("main_actions", "")
    if not actions_val:
        raw_actions = cap.get("actions", [])
        actions_val = "；".join(str(a) for a in raw_actions) if isinstance(raw_actions, list) else str(raw_actions)

    # 兼容 interaction 和 interactions
    interactions_val = cap.get("interactions", "") or cap.get("interaction", "")

    caption = {
        "time_of_day": cap.get("time_of_day", "未知"),
        "scene": cap.get("scene", "未知"),
        "space": cap.get("space", "未知"),
        "characters": cap.get("characters", []),
        "main_actions": actions_val,
        "interactions": interactions_val,
        "emotion": cap.get("emotion", ""),
    }

    return json.dumps(caption, ensure_ascii=False)


def _build_knowledge_base_by_rules(
    video_title: str,
    scenes: list[Scene],
) -> dict:
    """
    基于规则构建知识库 (无 LLM 时的回退方案)。

    规则:
    - scene/time_of_day 变化时创建新大事件
    - 每个场景 → 一个小事件
    """
    kb = {}
    current_act = None
    act_index = 0
    current_act_tod = "未知"
    current_act_scene_type = "未知"

    for scene in scenes:
        # 跳过片头曲/片尾曲
        if scene.content_type in ("opening", "ending"):
            continue

        cap = scene.structured_caption or {}
        tod = cap.get("time_of_day", "未知")
        scene_type = cap.get("scene", "未知")

        # 判断是否需要新大事件
        if current_act is None:
            act_index += 1
            current_act = f"第{act_index}幕: {tod}{scene_type}"
            kb[current_act] = {}
        else:
            # 时间或场景变化 → 新大事件
            if (tod != "未知" and current_act_tod != "未知" and tod != current_act_tod) or \
               (scene_type != "未知" and current_act_scene_type != "未知" and scene_type != current_act_scene_type):
                act_index += 1
                current_act = f"第{act_index}幕: {tod}{scene_type}"
                kb[current_act] = {}

        current_act_tod = tod
        current_act_scene_type = scene_type

        # 小事件
        actions_val = cap.get("main_actions", "")
        if not actions_val:
            raw_actions = cap.get("actions", [])
            actions_val = "；".join(str(a) for a in raw_actions) if isinstance(raw_actions, list) else str(raw_actions)

        characters = cap.get("characters", [])
        interactions_val = cap.get("interactions", "") or cap.get("interaction", "")

        if actions_val:
            sub_name = actions_val[:20]
        else:
            sub_name = f"场景{scene.index}"

        parts = []
        if characters:
            parts.append("、".join(str(c) for c in characters))
        if actions_val:
            parts.append(actions_val)
        if interactions_val:
            parts.append(interactions_val)
        desc = "；".join(parts) if parts else "无内容"

        kb[current_act][sub_name] = desc[:50]

    return {video_title: kb}
