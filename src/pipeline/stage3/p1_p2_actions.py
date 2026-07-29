"""Stage 3 P1 (Action 抽取) + P2 (Event 聚合).

从原 scripts/stage3_dryrun.py 抽出核心逻辑, 供 src.pipeline.stage3 编排。

输入: data/output/{video}/audio.json + visual.json
输出: data/output/{video}/stage3_dryrun.json (actions + events + stats + cost)

P1: 把 audio_segment + scene + ocr + caption 喂给 LLM, 抽 11 类 Communicative Action.
    batch_size 默认 5, 输出 results 数量必须等于 batch 大小 (否则重试).
P2: 同集按时间序排列的 Action 喂给 LLM, 聚合成 Event (含 participants / motivation /
    outcome / summary / retrieval_text / keywords + actions[].target).
"""

from __future__ import annotations

import json
import os
from collections import Counter

from src.core.config import get_config
from src.core.cost import get_cost_tracker, reset_cost_tracker
from src.core.helpers.json_utils import load_json, save_json
from src.core.llm.qwen_text import QwenTextClient
from src.core.logging import get_logger
from src.pipeline.stage3.llm_json import parse_json_with_repair

logger = get_logger()


# ============================================================
# 工具函数
# ============================================================

def parse_caption(caption_raw):
    """从 visual.captions[index] (可能是 ```json``` 字符串) 解析为 dict.

    失败时返回空骨架, 不抛异常。
    """
    empty = {
        "time_of_day": "",
        "scene": "",
        "space": "",
        "characters": [],
        "actions": [],
        "interaction": "",
        "temporal_relation": "",
    }
    if not caption_raw or not isinstance(caption_raw, str):
        return empty
    from src.core.helpers.text_utils import extract_json_obj
    obj = extract_json_obj(caption_raw)
    if isinstance(obj, dict):
        for k, v in empty.items():
            obj.setdefault(k, v)
        return obj
    return empty


def should_skip(scene: dict, seg: dict, ocr_text: str) -> tuple[bool, str]:
    """规则预过滤: 返回 (是否跳过, 原因)。无需 LLM。"""
    # 1. silence 锚点直接跳
    if scene.get("anchor_type") == "silence":
        return True, "silence_anchor"

    text = (seg.get("text") or "").strip()
    ocr_text = (ocr_text or "").strip()

    # 2. 完全无内容
    if ocr_text in ("", "无字幕") and not text:
        return True, "empty_content"

    # 3. ASR 太短且无 OCR (背景噪音/单字语气)
    if ocr_text in ("", "无字幕") and len(text) < 2:
        return True, "too_short_noise"

    # 4. 路人 + 极短发言
    speaker = seg.get("speaker_pred", "") or ""
    vp = seg.get("vp_score", 0) or 0
    is_pedestrian = (not speaker) or (vp < 0.4)
    if is_pedestrian and len(text) < 4:
        return True, "pedestrian_noise"

    return False, ""


def deduplicate_anchors(candidates: list[dict], scenes: list[dict]) -> list[dict]:
    """同 segment_id 只保留一个锚点: midpoint 优先于 switch, 否则保留首次出现。

    解决 stage2 同 segment 多锚点 (switch + midpoint) 导致 Action 重复的问题。
    """
    seen_segments: dict[str, int] = {}  # segment_id -> idx in deduped list
    deduped: list[dict] = []

    for c in candidates:
        if c["skip"]:
            continue
        scene = scenes[c["idx"]]
        seg_id = scene.get("segment_id")
        if not seg_id:
            deduped.append(c)
            continue

        anchor_type = scene.get("anchor_type", "")
        if seg_id not in seen_segments:
            seen_segments[seg_id] = len(deduped)
            deduped.append(c)
        else:
            existing = deduped[seen_segments[seg_id]]
            existing_scene = scenes[existing["idx"]]
            if existing_scene.get("anchor_type") != "midpoint" and anchor_type == "midpoint":
                deduped[seen_segments[seg_id]] = c
    return deduped


def build_p1_payload(video_id: str, audio: dict, visual: dict, batch_indices: list[int]) -> dict:
    """构建一个 P1 batch 的输入 payload。"""
    segments_by_id = {s["segment_id"]: s for s in audio.get("segments", [])}
    scenes = visual["scenes"]
    ocr_map = visual.get("ocr", {})
    captions_map = visual.get("captions", {})

    batch = []
    for idx in batch_indices:
        scene = scenes[idx]
        seg = segments_by_id.get(scene.get("segment_id"), {})
        ocr_text = ocr_map.get(str(idx), "无字幕")
        caption = parse_caption(captions_map.get(str(idx), ""))

        keyframe_path = ""
        kfps = scene.get("keyframe_paths") or []
        if kfps:
            keyframe_path = kfps[0]

        batch.append({
            "audio_segment": {
                "segment_id": seg.get("segment_id", ""),
                "begin_time": seg.get("begin_time", 0),
                "end_time": seg.get("end_time", 0),
                "speaker_pred": seg.get("speaker_pred", ""),
                "vp_score": seg.get("vp_score", 0),
                "text": seg.get("text", ""),
                "emotion": seg.get("emotion", ""),
            },
            "scene": {
                "scene_id": scene.get("scene_id", ""),
                "index": scene.get("index", 0),
                "start_time": scene.get("start_time", 0),
                "end_time": scene.get("end_time", 0),
                "anchor_type": scene.get("anchor_type", ""),
                "speaker": scene.get("speaker", ""),
                "anchor_text": scene.get("anchor_text", ""),
                "switch_from": scene.get("switch_from"),
                "switch_to": scene.get("switch_to"),
                "keyframe_path": keyframe_path,
            },
            "ocr": ocr_text,
            "caption": caption,
        })
    return {"video_id": video_id, "batch": batch}


# ============================================================
# LLM 调用 (P1 带数量校验, 自维护重试)
# ============================================================

def call_p1(client: QwenTextClient, prompt_template: str, payload: dict,
            stage: str = "stage3_p1", max_retries: int = 2) -> list:
    """调用 P1, 返回 results 列表。失败重试最多 max_retries 次。

    P1 强约束: results 数量必须等于 batch 大小, 否则视为失败并重试。
    """
    input_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = prompt_template.replace("__INPUT_BATCH_JSON__", input_json)
    expected = len(payload["batch"])
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            raw = client.generate(prompt=prompt, stage=stage, max_tokens=6000)
            if not raw:
                raise RuntimeError("LLM 返回空")
            parsed = parse_json_with_repair(raw)
            if not isinstance(parsed, dict) or "results" not in parsed:
                raise RuntimeError(f"返回非 JSON 或缺 results, 前 200 字: {raw[:200]!r}")
            if len(parsed["results"]) != expected:
                raise RuntimeError(f"results 数量不一致: 期望 {expected}, 实际 {len(parsed['results'])}")
            return parsed["results"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                logger.warning(f"          [RETRY {attempt+1}/{max_retries}] {e}")
                continue
    raise last_err


def call_p2(client: QwenTextClient, prompt_template: str, payload: dict,
            stage: str = "stage3_p2", max_retries: int = 2) -> dict:
    """调用 P2, 返回 {events, characters_to_resolve}。失败重试最多 max_retries 次。"""
    input_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = prompt_template.replace("__INPUT_ACTIONS_JSON__", input_json)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            raw = client.generate(prompt=prompt, stage=stage, max_tokens=6000)
            if not raw:
                raise RuntimeError("LLM 返回空")
            parsed = parse_json_with_repair(raw)
            if not isinstance(parsed, dict) or "events" not in parsed:
                raise RuntimeError(f"返回非 JSON 或缺 events, 前 200 字: {raw[:200]!r}")
            return parsed
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                logger.warning(f"          [RETRY {attempt+1}/{max_retries}] {e}")
                continue
    raise last_err


# ============================================================
# 主流程
# ============================================================

def run_p1p2(
    video_dir: str,
    p1_batch: int = 5,
    p2_batch: int = 8,
    limit_p1: int = 0,
    save: bool = True,
) -> dict:
    """跑 P1 + P2, 返回 stage3_dryrun 格式的 dict.

    Args:
        video_dir: 视频目录名 (相对 output_root)
        p1_batch: P1 batch 大小 (候选数/批)
        p2_batch: P2 batch 大小 (action 数/批)
        limit_p1: 只跑前 N 个候选 (0=全部), 用于快速验证
        save: 是否写入 data/output/{video}/stage3_dryrun.json

    Returns:
        {video_id, stats, cost, actions, events, characters_to_resolve}
    """
    reset_cost_tracker()

    config = get_config()
    output_dir = os.path.join(config.output_root, video_dir)
    audio_path = os.path.join(output_dir, "audio.json")
    visual_path = os.path.join(output_dir, "visual.json")

    if not os.path.isfile(audio_path) or not os.path.isfile(visual_path):
        raise FileNotFoundError(
            f"缺少 audio.json 或 visual.json: {output_dir}; 请先跑 Stage 1/2"
        )

    audio = load_json(audio_path)
    visual = load_json(visual_path)
    logger.info(f"[INFO] video_id = {video_dir}")
    logger.info(f"[INFO] 加载 audio.json: {len(audio.get('segments', []))} segments")
    logger.info(f"[INFO] 加载 visual.json: {len(visual.get('scenes', []))} scenes, "
                f"{sum(1 for v in visual.get('ocr', {}).values() if v and v != '无字幕')} 条非空 OCR, "
                f"{len(visual.get('captions', {}))} 条 caption")

    # 加载 prompt
    p1_prompt = config.prompts["stage3_p1_action_extract"]["user"]
    p2_prompt = config.prompts["stage3_p2_event_aggregate"]["user"]

    # 客户端
    client = QwenTextClient()

    # ---------- 1. 规则预过滤 ----------
    segments_by_id = {s["segment_id"]: s for s in audio.get("segments", [])}
    scenes = visual["scenes"]
    ocr_map = visual.get("ocr", {})

    candidates = []
    for scene in scenes:
        idx = scene["index"]
        seg = segments_by_id.get(scene.get("segment_id"), {})
        ocr_text = ocr_map.get(str(idx), "无字幕")
        skip, reason = should_skip(scene, seg, ocr_text)
        candidates.append({"idx": idx, "skip": skip, "reason": reason})

    n_total = len(candidates)
    n_skipped = sum(1 for c in candidates if c["skip"])
    skip_rate = n_skipped * 100 / n_total if n_total else 0
    logger.info(f"[STEP 1] 规则预过滤: 跳过 {n_skipped}/{n_total} ({skip_rate:.1f}%)")
    reason_counts = Counter(c["reason"] for c in candidates if c["skip"])
    for r, n in reason_counts.most_common():
        logger.info(f"          {r}: {n}")

    # ---------- 1.5 anchor 去重 ----------
    deduped_candidates = deduplicate_anchors(candidates, scenes)
    n_after_dedup = len(deduped_candidates)
    n_dup_removed = (n_total - n_skipped) - n_after_dedup
    logger.info(f"[STEP 1.5] anchor 去重: 移除 {n_dup_removed} 个同 segment 重复锚点, "
                f"剩余 {n_after_dedup} 候选")

    # ---------- 2. P1 Batch 抽 Action ----------
    unsent_indices = [c["idx"] for c in deduped_candidates]
    if limit_p1 > 0:
        unsent_indices = unsent_indices[:limit_p1]
        logger.info(f"[STEP 2] --limit-p1={limit_p1}, 只跑前 {len(unsent_indices)} 个候选")

    p1_batches = [unsent_indices[i:i + p1_batch]
                  for i in range(0, len(unsent_indices), p1_batch)]
    logger.info(f"[STEP 2] P1: {len(unsent_indices)} 候选 → {len(p1_batches)} 批 (batch_size={p1_batch})")

    actions = []
    p1_errors = 0
    for bi, batch_indices in enumerate(p1_batches, 1):
        payload = build_p1_payload(video_dir, audio, visual, batch_indices)
        try:
            results = call_p1(client, p1_prompt, payload)
        except Exception as e:
            logger.error(f"[ERROR] P1 batch {bi} 失败: {e}")
            p1_errors += 1
            continue

        if len(results) != len(batch_indices):
            logger.warning(f"[WARN] P1 batch {bi}: 输入 {len(batch_indices)} 条, "
                           f"输出 {len(results)} 条, 数量不一致")

        for result in results:
            if result.get("skip"):
                continue
            action = result.get("action")
            if action:
                actions.append(action)

        logger.info(f"          batch {bi}/{len(p1_batches)} 完成, 累计 {len(actions)} actions")

    logger.info(f"[STEP 2] P1 总计: {len(actions)} actions ({p1_errors} 批失败)")

    # ---------- 3. P2 聚 Event ----------
    p2_batches = [actions[i:i + p2_batch]
                  for i in range(0, len(actions), p2_batch)]
    logger.info(f"[STEP 3] P2: {len(actions)} actions → {len(p2_batches)} 批 "
                f"(batch_size={p2_batch})")

    events = []
    characters_to_resolve = []
    p2_errors = 0
    for bi, batch in enumerate(p2_batches, 1):
        payload = {
            "video_id": video_dir,
            "video_title": video_dir,
            "actions": batch,
            "characters_known": [],  # P1+P2 阶段不加载全局 characters
        }
        try:
            result = call_p2(client, p2_prompt, payload)
        except Exception as e:
            logger.error(f"[ERROR] P2 batch {bi} 失败: {e}")
            p2_errors += 1
            continue

        batch_events = result.get("events", [])
        events.extend(batch_events)
        characters_to_resolve.extend(result.get("characters_to_resolve", []))
        logger.info(f"          batch {bi}/{len(p2_batches)} 完成, 累计 {len(events)} events")

    logger.info(f"[STEP 3] P2 总计: {len(events)} events, "
                f"{len(characters_to_resolve)} 个待归一角色 ({p2_errors} 批失败)")

    # ---------- 3.5 event_id 跨 batch 重编号 ----------
    for i, e in enumerate(events):
        e["event_id"] = f"{video_dir}_e{i:03d}"
    logger.info(f"[STEP 3.5] event_id 跨 batch 重编号完成: e000–e{len(events)-1:03d}")

    # Action 类型分布
    action_dist = Counter(a.get("action", "?") for a in actions)
    mc_dist = Counter(e.get("motivation_confidence", "?") for e in events)

    # ---------- 4. 保存 ----------
    result_data = {
        "video_id": video_dir,
        "stats": {
            "n_scenes_total": n_total,
            "n_skipped_by_rule": n_skipped,
            "skip_rate": round(skip_rate, 2),
            "skip_reasons": dict(reason_counts),
            "n_dup_anchors_removed": n_dup_removed,
            "n_candidates_after_dedup": n_after_dedup,
            "n_p1_calls": len(p1_batches),
            "n_p1_errors": p1_errors,
            "n_p2_calls": len(p2_batches),
            "n_p2_errors": p2_errors,
            "n_actions": len(actions),
            "n_events": len(events),
            "action_type_dist": dict(action_dist),
            "motivation_confidence_dist": dict(mc_dist),
        },
        "cost": get_cost_tracker().report_dict(),
        "actions": actions,
        "events": events,
        "characters_to_resolve": characters_to_resolve,
    }

    if save:
        out_path = os.path.join(output_dir, "stage3_dryrun.json")
        save_json(result_data, out_path)
        logger.info(f"[INFO] 结果已保存: {out_path}")

    # Cost 报告
    tracker = get_cost_tracker()
    logger.info("=" * 70)
    logger.info("Cost 报告")
    logger.info("=" * 70)
    logger.info(tracker.report())

    return result_data
