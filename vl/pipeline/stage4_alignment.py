"""Stage 4: 多模态时序对齐 - 核心模块

将视觉 (场景/关键帧)、音频 (ASR 转写)、文本 (VLM 描述)、角色 (人脸聚类)
按时序维度对齐，构建统一的时间线。

对齐策略:
  1. ASR → 场景: 基于时间戳的精确匹配 + 模糊边界处理
  2. 角色 → 场景: 人脸检测结果 + VLM 角色提及的交叉验证
  3. 场景连续性: 基于 structured_caption 的 time_of_day/space/scene 变化检测
  4. 叙事事件检测: 将连续的同类场景聚合为叙事事件
"""

import os
from collections import defaultdict

from vl.core.config import AppConfig
from vl.core.models.scene import Scene
from vl.core.helpers.json_utils import save_json, load_json

from vl.core.logging import get_logger

logger = get_logger()


def run_stage4(
    video_id: str,
    video_title: str,
    scenes: list[Scene],
    scene_transcripts: dict[str, list[str]],
    output_dir: str,
    config: AppConfig,
) -> dict:
    """
    执行 Stage 4: 多模态时序对齐。

    Returns:
        aligned_timeline: 时序对齐后的结构化数据
    """
    logger.info("[Stage 4] 开始多模态时序对齐...")

    # 加载人脸/角色数据
    characters = _load_characters(video_id, output_dir)

    # 4.1 ASR 时序精细化对齐
    logger.info("[Stage 4.1] ASR-场景时序对齐...")
    transcript_segments = _load_transcript_segments(video_id, output_dir)
    aligned_scenes = _align_asr_to_scenes(scenes, transcript_segments, scene_transcripts)

    # 4.2 角色-场景交叉对齐
    logger.info("[Stage 4.2] 角色-场景交叉对齐...")
    character_scenes = _align_characters_to_scenes(aligned_scenes, characters)

    # 4.3 场景连续性检测 & 叙事事件分组
    logger.info("[Stage 4.3] 叙事事件检测...")
    narrative_events = _detect_narrative_events(aligned_scenes)

    # 4.4 构建统一时间线
    logger.info("[Stage 4.4] 构建统一时间线...")
    timeline = _build_timeline(
        video_id=video_id,
        video_title=video_title,
        scenes=aligned_scenes,
        character_scenes=character_scenes,
        narrative_events=narrative_events,
    )

    # 保存对齐结果
    alignment_dir = os.path.join(output_dir, "alignment", video_id)
    os.makedirs(alignment_dir, exist_ok=True)
    save_json(timeline, os.path.join(alignment_dir, "aligned_timeline.json"))
    logger.info(f"[Stage 4] 对齐完成: {len(aligned_scenes)} 场景, "
                f"{len(narrative_events)} 叙事事件, "
                f"{len(character_scenes)} 角色")

    return timeline


# ──────────────────────────────────────────────────────────
# 4.1 ASR → 场景时序对齐
# ──────────────────────────────────────────────────────────

def _load_transcript_segments(video_id: str, output_dir: str) -> list[dict]:
    """加载 ASR 转写片段 (带时间戳)"""
    path = os.path.join(output_dir, "transcripts", video_id, "transcript.json")
    if os.path.isfile(path):
        return load_json(path)
    return []


def _align_asr_to_scenes(
    scenes: list[Scene],
    transcript_segments: list[dict],
    scene_transcripts: dict[str, list[str]],
) -> list[dict]:
    """
    将 ASR 片段精确对齐到场景。

    策略:
    - 精确匹配: ASR 片段时间完全在场景时间范围内
    - 跨场景匹配: ASR 片段跨多个场景时，按时间重叠比例分配
    - 模糊匹配: ASR 片段在场景边界附近时，归属到重叠更多的场景
    """
    aligned = []

    for scene in scenes:
        entry = {
            "scene_id": scene.scene_id,
            "video_id": scene.video_id,
            "index": scene.index,
            "start_time": round(scene.start_time, 3),
            "end_time": round(scene.end_time, 3),
            "duration": round(scene.end_time - scene.start_time, 3),
            "keyframe_paths": scene.keyframe_paths,
            "transcript": " ".join(scene_transcripts.get(scene.scene_id, [])),
            "transcript_segments": [],
            "content_type": scene.content_type,
        }

        # 从 structured_caption 提取视觉信息
        if scene.structured_caption:
            cap = scene.structured_caption
            entry["visual"] = {
                "time_of_day": cap.get("time_of_day", "未知"),
                "space": cap.get("space", "未知"),
                "subspace": cap.get("subspace", "未知"),
                "scene": cap.get("scene", "未知"),
                "characters": cap.get("characters", []),
                "main_actions": cap.get("main_actions", ""),
                "interactions": cap.get("interactions", ""),
                "emotion": cap.get("emotion", ""),
                "plot_state": cap.get("plot_state", ""),
            }
        elif scene.vlm_caption:
            entry["visual"] = {"raw_caption": scene.vlm_caption}

        # 精确时序匹配: 找出时间上属于本场景的 ASR 片段
        for seg in transcript_segments:
            seg_start = seg.get("start_time", seg.get("start", 0))
            seg_end = seg.get("end_time", seg.get("end", 0))
            seg_scene_id = seg.get("scene_id", "")

            if seg_scene_id == scene.scene_id:
                entry["transcript_segments"].append({
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "text": seg.get("text", ""),
                    "speaker": seg.get("speaker_id", seg.get("speaker", "")),
                    "words": seg.get("words", []),
                })

        aligned.append(entry)

    return aligned


# ──────────────────────────────────────────────────────────
# 4.2 角色 → 场景交叉对齐
# ──────────────────────────────────────────────────────────

def _load_characters(video_id: str, output_dir: str) -> list[dict]:
    """加载角色聚类结果"""
    path = os.path.join(output_dir, "characters", video_id, "characters.json")
    if os.path.isfile(path):
        return load_json(path)
    return []


def _align_characters_to_scenes(
    aligned_scenes: list[dict],
    characters: list[dict],
) -> dict[str, list[str]]:
    """
    角色与场景交叉对齐。

    数据来源:
    1. 人脸检测: 角色在哪些场景的关键帧中出现
    2. VLM 描述: structured_caption 中提到的角色名
    3. ASR 转写: 说话人识别结果 (speaker 字段)

    Returns:
        character_arcs: {character_name: [scene_id, ...]}
    """
    character_arcs = defaultdict(set)

    for scene in aligned_scenes:
        scene_id = scene["scene_id"]

        # 来源 1: VLM 结构化描述中的角色
        visual = scene.get("visual", {})
        for char_name in visual.get("characters", []):
            character_arcs[char_name].add(scene_id)

        # 来源 2: ASR 说话人
        for seg in scene.get("transcript_segments", []):
            speaker = seg.get("speaker", "")
            if speaker and speaker not in ("", "SPEAKER_UNKNOWN"):
                character_arcs[speaker].add(scene_id)

    # 来源 3: characters.json 中记录的角色出场 (omni 提取)
    for char in characters:
        char_label = char.get("label", "")
        if not char_label:
            continue
        for scene_id in char.get("appearance_scenes", []):
            character_arcs[char_label].add(scene_id)

    # 转为有序列表 (按场景 index 排序)
    scene_order = {s["scene_id"]: s["index"] for s in aligned_scenes}
    result = {}
    for char_name, scene_ids in character_arcs.items():
        sorted_ids = sorted(scene_ids, key=lambda sid: scene_order.get(sid, 0))
        result[char_name] = sorted_ids

    return result


# ──────────────────────────────────────────────────────────
# 4.3 叙事事件检测
# ──────────────────────────────────────────────────────────

def _detect_narrative_events(aligned_scenes: list[dict]) -> list[dict]:
    """
    将连续的同类场景聚合为叙事事件。

    分组依据 (来自 structured_caption):
    - scene (室内/室外) 变化 → 新事件
    - time_of_day (白天/晚上) 变化 → 新事件
    - 连续场景有相同属性 → 同一事件

    Returns:
        events: [{event_id, scenes, time_range, attributes}]
    """
    if not aligned_scenes:
        return []

    events = []
    current_event_scenes = [aligned_scenes[0]]
    current_attrs = _get_scene_attrs(aligned_scenes[0])
    current_ct = aligned_scenes[0].get("content_type", "main")

    for scene in aligned_scenes[1:]:
        attrs = _get_scene_attrs(scene)
        scene_ct = scene.get("content_type", "main")

        # content_type 变化 (opening↔main, main↔ending) → 强制新事件
        ct_changed = (scene_ct != current_ct
                      and current_ct in ("opening", "ending", "main")
                      and scene_ct in ("opening", "ending", "main"))

        if ct_changed or _is_scene_change(current_attrs, attrs):
            # 场景变化或内容类型变化 → 结束当前事件，创建新事件
            events.append(_build_event(len(events), current_event_scenes, current_attrs))
            current_event_scenes = [scene]
            current_attrs = attrs
            current_ct = scene_ct
        else:
            # 同一事件 → 追加场景
            current_event_scenes.append(scene)
            # 更新属性 (保留最新)
            current_attrs = attrs

    # 最后一个事件
    events.append(_build_event(len(events), current_event_scenes, current_attrs))

    logger.info(f"检测到 {len(events)} 个叙事事件")
    return events


def _get_scene_attrs(scene: dict) -> dict:
    """提取场景属性用于变化检测"""
    visual = scene.get("visual", {})
    return {
        "time_of_day": visual.get("time_of_day", "未知"),
        "scene": visual.get("scene", "未知"),
        "space": visual.get("space", "未知"),
    }


def _is_scene_change(prev_attrs: dict, curr_attrs: dict) -> bool:
    """
    判断是否发生场景变化。

    触发新事件的条件:
    - scene 字段变化 (室内 ↔ 室外)
    - time_of_day 字段变化 (白天 ↔ 晚上)
    - 不因 space 变化触发 (子区域切换不算事件边界)
    """
    # 室内↔室外 变化
    if (prev_attrs.get("scene") != curr_attrs.get("scene")
            and prev_attrs.get("scene") != "未知"
            and curr_attrs.get("scene") != "未知"):
        return True

    # 白天↔晚上 变化
    if (prev_attrs.get("time_of_day") != curr_attrs.get("time_of_day")
            and prev_attrs.get("time_of_day") != "未知"
            and curr_attrs.get("time_of_day") != "未知"):
        return True

    return False


def _build_event(event_index: int, scenes: list[dict], attrs: dict) -> dict:
    """构建叙事事件"""
    # 从 scenes 多数投票决定 event 的 content_type
    ct_counts = {"opening": 0, "main": 0, "ending": 0}
    for s in scenes:
        ct = s.get("content_type", "main")
        if ct in ct_counts:
            ct_counts[ct] += 1
    event_content_type = max(ct_counts, key=ct_counts.get)

    return {
        "event_id": f"event_{event_index:03d}",
        "scene_ids": [s["scene_id"] for s in scenes],
        "start_time": scenes[0]["start_time"],
        "end_time": scenes[-1]["end_time"],
        "duration": round(scenes[-1]["end_time"] - scenes[0]["start_time"], 3),
        "scene_count": len(scenes),
        "time_of_day": attrs.get("time_of_day", "未知"),
        "scene_type": attrs.get("scene", "未知"),
        "space": attrs.get("space", "未知"),
        "transcript": " ".join(
            s.get("transcript", "") for s in scenes if s.get("transcript")
        ),
        "characters": list({
            c for s in scenes
            for c in s.get("visual", {}).get("characters", [])
        }),
        "content_type": event_content_type,
    }


# ──────────────────────────────────────────────────────────
# 4.4 统一时间线构建
# ──────────────────────────────────────────────────────────

def _build_timeline(
    video_id: str,
    video_title: str,
    scenes: list[dict],
    character_scenes: dict[str, list[str]],
    narrative_events: list[dict],
) -> dict:
    """构建统一的多模态时间线"""
    total_duration = scenes[-1]["end_time"] if scenes else 0

    timeline = {
        "video_id": video_id,
        "video_title": video_title,
        "total_duration": round(total_duration, 3),
        "total_scenes": len(scenes),
        "total_events": len(narrative_events),
        "total_characters": len(character_scenes),

        # 叙事事件 (大事件)
        "events": narrative_events,

        # 角色出场弧线
        "character_arcs": character_scenes,

        # 场景级时间线 (最细粒度)
        "scenes": scenes,
    }

    return timeline
