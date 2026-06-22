"""
Stage 2: 视觉处理

流程 (声纹锚点驱动):
  1. 从 audio_result 的声纹 segments 生成关键帧锚点 (midpoint / switch / silence)
  2. 按锚点时间戳采样帧 — 每帧天然绑定 speaker + 台词
  3. Qwen-VL-Max 字幕 OCR (聚焦画面底部字幕行)
  4. Qwen-VL-Max 视觉描述生成 (用 anchor 台词作为强上下文)
  → scenes.json + keyframes/ + visual.json

无 audio_result 时回退到旧的 SBD + 均匀采样路径.
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np

from src.core.helpers.json_utils import save_json
from src.core.helpers.prompt_loader import load_prompt
from src.core.config import get_config
from src.core.logging import get_logger
from src.scene import create_detector
from src.scene.speaker_anchor import Anchor, build_anchors

logger = get_logger()


# 字幕 OCR 专用 prompt — 抽帧前已裁剪到底部 25% 字幕区域, 直接问"图中字幕"
SUBTITLE_OCR_PROMPT = (
    "请提取这张图片中的字幕文字 (角色台词的硬字幕). "
    "只输出字幕原文; 若没有字幕, 仅回复: 无字幕."
)


def _imwrite_unicode(path: str, frame) -> bool:
    """cv2.imwrite 在 Windows 中文路径下会失败, 用 imencode + tofile 替代"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, frame)
    if ok:
        buf.tofile(path)
    return ok


def _crop_bottom_25(frame):
    """裁剪底部 25% 字幕区域 — 减少 OCR 输入 token, 同时排除画面内其他文字干扰."""
    h = frame.shape[0]
    return frame[int(h * 0.75):, :]


def _anchor_to_scene(anchor: Anchor, idx: int, video_id: str, fps: float) -> dict:
    """把单个锚点转成 scenes.json 兼容的 scene dict.

    每个锚点等价于一个"语义场景单元":
      - midpoint: 时间窗口 = 源 segment 的 [begin, end]
      - switch:   时间窗口 = (t-0.5, t+0.5), 捕获反应镜头
      - silence:  时间窗口 = (t-1.0, t+1.0), 兜底无对话段
    """
    t = anchor.timestamp
    if anchor.anchor_type == "midpoint":
        # 源 segment 时间窗口; segment_id 可索引回 audio.json
        # 这里用 ±0.5s 兜底, 真实窗口应由调用方注入; 简化处理: ±0.5s
        s, e = t - 0.5, t + 0.5
    elif anchor.anchor_type == "switch":
        s, e = t - 0.5, t + 0.5
    else:  # silence
        s, e = t - 1.0, t + 1.0

    return {
        "scene_id": f"{video_id}_a{idx:04d}",
        "video_id": video_id,
        "index": idx,
        "start_time": round(s, 3),
        "end_time": round(e, 3),
        "start_frame": int(max(0, s * fps)) if fps > 0 else 0,
        "end_frame": int(max(0, e * fps)) if fps > 0 else 0,
        "keyframe_paths": [],
        "transition_type": anchor.anchor_type,  # 复用字段存锚点类型
        "vlm_caption": None,
        "structured_caption": None,
        "content_type": "main",
        "confidence": 1.0,
        # 锚点扩展字段
        "anchor_type": anchor.anchor_type,
        "anchor_timestamp": round(t, 3),
        "speaker": anchor.speaker,
        "segment_id": anchor.segment_id,
        "anchor_text": anchor.text,
        "switch_from": anchor.switch_from,
        "switch_to": anchor.switch_to,
    }


def extract_anchor_keyframes(video_path, anchors, output_dir, video_id):
    """按声纹锚点采样关键帧, 每个锚点产 1 帧, 返回 scenes 列表.

    同一 frame_idx 上的多个锚点共享同一张图 (硬链接/复制), 避免重复解码.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    scenes: list[dict] = []
    # frame_idx → 已写入文件名, 避免重复解码与重复写盘
    frame_cache: dict[int, str] = {}

    for idx, a in enumerate(anchors):
        scene = _anchor_to_scene(a, idx, video_id, fps)
        if fps <= 0:
            scenes.append(scene)
            continue
        frame_idx = int(a.timestamp * fps)
        if frame_idx >= total_frames:
            frame_idx = max(0, total_frames - 1)

        if frame_idx in frame_cache:
            path = frame_cache[frame_idx]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                scenes.append(scene)
                continue
            spk_slug = (a.speaker or "anon").replace(" ", "_")
            name = f"a{idx:04d}_{a.anchor_type}_{a.timestamp:07.2f}s_{spk_slug}.jpg"
            path = os.path.join(keyframes_dir, name)
            _imwrite_unicode(path, frame)
            frame_cache[frame_idx] = path

        scene["keyframe_paths"] = [path]
        scenes.append(scene)

    cap.release()
    return scenes


# ---- 旧路径 (无 audio_result 时回退) -----------------------------------------

def extract_keyframes(video_path, scenes, output_dir, samples_per_scene=8):
    """[legacy] 为每个场景均匀采样关键帧. 仅在缺少声纹 segments 时使用."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    for scene in scenes:
        start_frame = scene["start_frame"]
        end_frame = scene["end_frame"]
        scene_idx = scene["index"]
        n_frames = end_frame - start_frame

        if n_frames <= 0:
            continue

        # 均匀采样
        if n_frames <= samples_per_scene:
            indices = list(range(start_frame, end_frame))
        else:
            indices = np.linspace(start_frame, end_frame - 1, samples_per_scene, dtype=int)

        for j, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(keyframes_dir, f"scene_{scene_idx:04d}_frame_{j:02d}.jpg")
                _imwrite_unicode(path, frame)
                scene.setdefault("keyframe_paths", []).append(path)

    cap.release()
    return scenes


def filter_subtitle_frames(scenes, audio_segments=None):
    """[legacy] 保留首+中+末帧. 仅旧路径使用."""
    for scene in scenes:
        kf = scene.get("keyframe_paths", [])
        if len(kf) <= 2:
            continue

        mid = len(kf) // 2
        kept = [kf[0], kf[mid], kf[-1]] if len(kf) > 2 else kf

        for p in kf:
            if p not in kept and os.path.isfile(p):
                os.remove(p)

        scene["keyframe_paths"] = kept

    return scenes


def run_ocr(scenes, config, video_path: str | None = None, max_workers: int = 8):
    """字幕 OCR — 两阶段 tiered 策略, 最优 cost / accuracy / speed 平衡.

    设计原则:
      - 只对 midpoint 锚点跑 OCR (switch 锚点的字幕信息已被同段 midpoint 覆盖)
      - silence 锚点直接标 '无字幕' (B-roll 无对白)
      - switch 锚点继承其 segment_id 对应 midpoint 的结果
      - Phase 1: 先 OCR 中心帧, 大多数 (~80%) 一次命中, 早退
      - Phase 2: 仅对中心帧 miss 的锚点采样 4 个邻帧 [-0.5,-0.25,+0.25,+0.5]
      - 并发执行 (默认 8 workers)

    Args:
        scenes: 已带 anchor_timestamp / anchor_type / segment_id 的 scene 列表
        config: AppConfig
        video_path: 视频路径; None 时回退到 keyframe_paths[0] 单帧模式
        max_workers: 并发线程数

    Returns:
        dict[int, str]: scene_index → 字幕文本 或 '无字幕'

    实测 (家有儿女 第001集, 22min):
      命中率: midpoint 95.4%, 非静默 ~95%
      总调用: ~470 (vs 全采样 2695)
      费用:   ~1.1 CNY
      耗时:   ~3 min
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.core.llm.qwen_vl import QwenVLClient

    ocr_model = config.model_ocr or "qwen3-vl-plus"
    NO_SUB = "无字幕"
    NEIGHBOR_OFFSETS = [-0.5, -0.25, 0.25, 0.5]  # 不含 0 (中心已跑过)

    # ===== 1. 分类锚点 =====
    # SBD 回退路径产出的 scene 没有 anchor_type — 统一当作 midpoint 处理,
    # 并补上 anchor_timestamp = 场景中点, 供下游 _sample_frame 使用.
    for s in scenes:
        if "anchor_type" not in s:
            s["anchor_type"] = "midpoint"
        if "anchor_timestamp" not in s:
            s["anchor_timestamp"] = round((s.get("start_time", 0) + s.get("end_time", 0)) / 2, 3)

    midpoint_scenes = [s for s in scenes if s.get("anchor_type") == "midpoint"]
    silence_indices = {s["index"] for s in scenes if s.get("anchor_type") == "silence"}
    switch_scenes = [s for s in scenes if s.get("anchor_type") == "switch"]

    ocr_results: dict[int, str] = {idx: NO_SUB for idx in silence_indices}

    # ===== 2. 单帧回退 (无视频路径) =====
    if not video_path or cv2 is None:
        import tempfile
        vl_client = QwenVLClient(model=ocr_model)
        tmp = tempfile.mkdtemp(prefix="videolens_ocr_crop_")
        for s in midpoint_scenes:
            kf = s.get("keyframe_paths", [])
            if not kf or not os.path.isfile(kf[0]): continue
            try:
                frame = cv2.imdecode(np.fromfile(kf[0], dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raw = vl_client.analyze_image(kf[0], SUBTITLE_OCR_PROMPT, stage="stage2_ocr")
                else:
                    crop_path = os.path.join(tmp, f"{s['index']:04d}.jpg")
                    _imwrite_unicode(crop_path, _crop_bottom_25(frame))
                    raw = vl_client.analyze_image(crop_path, SUBTITLE_OCR_PROMPT, stage="stage2_ocr")
                ocr_results[s["index"]] = (raw or "").strip() or NO_SUB
            except Exception as e:
                logger.warning(f"OCR midpoint {s['index']} 失败: {e}")
        _inherit_switch_from_midpoint(switch_scenes, midpoint_scenes, ocr_results)
        return ocr_results

    # ===== 3. 多帧 tiered 路径 =====
    import tempfile, shutil
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp_dir = tempfile.mkdtemp(prefix="videolens_ocr_")

    def _sample_frame(t: float, idx: int, off: float) -> str | None:
        """在时间 t 采样一帧, 裁剪底部 25% 后写入 tmp_dir, 返回路径"""
        if fps <= 0: return None
        t = max(0.0, t)
        fi = int(t * fps)
        if fi >= total_frames: fi = max(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: return None
        frame = _crop_bottom_25(frame)
        path = os.path.join(tmp_dir, f"{idx:04d}_{off:+.2f}.jpg")
        _imwrite_unicode(path, frame)
        return path

    # Phase 1: 所有 midpoint 的中心帧
    phase1_tasks: list[tuple[int, float, str]] = []
    try:
        for s in midpoint_scenes:
            t = float(s.get("anchor_timestamp", 0.0))
            p = _sample_frame(t, s["index"], 0.0)
            if p: phase1_tasks.append((s["index"], 0.0, p))
    finally:
        pass  # cap 暂不释放, Phase 2 还要用

    logger.info(f"OCR Phase 1: {len(phase1_tasks)} midpoint 中心帧")

    def _ocr_one(task):
        idx, off, path = task
        client = QwenVLClient(model=ocr_model)
        try:
            raw = client.analyze_image(path, SUBTITLE_OCR_PROMPT, stage="stage2_ocr")
            return idx, off, (raw or "").strip()
        except Exception as e:
            logger.warning(f"OCR scene={idx} off={off} 失败: {e}")
            return idx, off, ""

    phase1_results: dict[int, str] = {}
    misses: list[dict] = []  # Phase 2 候选
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_ocr_one, t) for t in phase1_tasks]
        for fut in as_completed(futs):
            idx, off, text = fut.result()
            phase1_results[idx] = text
            if not text or text == NO_SUB:
                # 找回原 scene 用于 Phase 2
                misses.append(next(s for s in midpoint_scenes if s["index"] == idx))

    n_p1_hit = sum(1 for t in phase1_results.values() if t and t != NO_SUB)
    logger.info(f"OCR Phase 1 命中: {n_p1_hit}/{len(phase1_results)}")

    # Phase 2: 仅对 miss 采样 4 个邻帧
    phase2_tasks: list[tuple[int, float, str]] = []
    try:
        for s in misses:
            t_center = float(s.get("anchor_timestamp", 0.0))
            for off in NEIGHBOR_OFFSETS:
                p = _sample_frame(t_center + off, s["index"], off)
                if p: phase2_tasks.append((s["index"], off, p))
    finally:
        cap.release()

    logger.info(f"OCR Phase 2: {len(misses)} miss × 4 邻帧 = {len(phase2_tasks)} 帧")

    phase2_results: dict[int, list[tuple[float, str]]] = {}
    if phase2_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_ocr_one, t) for t in phase2_tasks]
            for fut in as_completed(futs):
                idx, off, text = fut.result()
                phase2_results.setdefault(idx, []).append((off, text))

    # 清理临时目录
    try: shutil.rmtree(tmp_dir)
    except OSError: pass

    # ===== 4. 聚合 midpoint 结果 =====
    for s in midpoint_scenes:
        idx = s["index"]
        center = phase1_results.get(idx, "")
        if center and center != NO_SUB:
            ocr_results[idx] = center
            continue
        neighbors = phase2_results.get(idx, [])
        if neighbors:
            ocr_results[idx] = _aggregate_ocr(neighbors)
        else:
            ocr_results[idx] = NO_SUB

    # ===== 5. switch 继承同段 midpoint 结果 =====
    _inherit_switch_from_midpoint(switch_scenes, midpoint_scenes, ocr_results)

    return ocr_results


def _inherit_switch_from_midpoint(switch_scenes, midpoint_scenes, ocr_results):
    """switch 锚点继承其 segment_id 对应 midpoint 锚点的 OCR 结果.

    每个 switch 锚点的 segment_id 指向新 speaker 的 segment,
    该 segment 同时也产出了一个 midpoint 锚点, OCR 信息已被采集.
    """
    # 建 segment_id → midpoint ocr 映射
    seg_to_ocr: dict[str | None, str] = {}
    for s in midpoint_scenes:
        sid = s.get("segment_id")
        if sid is not None and sid not in seg_to_ocr:
            seg_to_ocr[sid] = ocr_results.get(s["index"], "无字幕")

    for s in switch_scenes:
        sid = s.get("segment_id")
        ocr_results[s["index"]] = seg_to_ocr.get(sid, "无字幕")


def _aggregate_ocr(frame_results: list[tuple[float, str]]) -> str:
    """聚合多帧 OCR 结果.

    规则:
      1. 过滤 '无字幕' / 空串
      2. 若剩余 0 条 → '无字幕'
      3. 若剩余多条 → 取最长的 (字幕最完整); 同长时优先 offset=0 的中心帧
    """
    NO_SUB = "无字幕"
    valid = [(off, t) for off, t in frame_results if t and t != NO_SUB]
    if not valid:
        return NO_SUB
    # 最长优先, 同长时偏移绝对值小者优先 (中心帧优先)
    valid.sort(key=lambda x: (-len(x[1]), abs(x[0])))
    return valid[0][1]


def run_captions(scenes, audio_result, config):
    """为每个关键帧生成视觉描述.

    优先使用锚点自带的 anchor_text 作为强上下文; 旧路径回退到时间窗口匹配.
    """
    from src.core.llm.qwen_vl import QwenVLClient

    vl_client = QwenVLClient()
    user_tpl, _ = load_prompt(config, "stage2_scene_caption")

    audio_segs = audio_result.get("segments", []) if audio_result else []

    captions = {}
    for scene in scenes:
        kf = scene.get("keyframe_paths", [])
        if not kf:
            continue

        # 锚点路径: 直接用 anchor_text; 旧路径: 时间窗口匹配
        anchor_text = (scene.get("anchor_text") or "").strip()
        if anchor_text:
            scene_text = anchor_text
        else:
            s, e = scene["start_time"], scene["end_time"]
            scene_text = " ".join(
                seg["text"] for seg in audio_segs
                if s <= seg.get("begin_time", 0) <= e or s <= seg.get("end_time", 0) <= e
            )

        first_kf = kf[0]
        if not os.path.isfile(first_kf):
            continue

        prompt = user_tpl.format(
            audiotext=scene_text[:200] if scene_text else "无",
            caption="",
        )

        try:
            result = vl_client.analyze_image(
                first_kf,
                prompt if prompt else "请简要描述这个画面的内容。",
                stage="stage2_caption",
            )
            if result:
                captions[scene["index"]] = result
        except Exception as e:
            logger.warning(f"Caption 场景 {scene['index']} 失败: {e}")

    return captions


def run_stage2(video_dir: str, output_dir: str, audio_result: dict = None, skip_captions: bool = False) -> dict:
    """Stage 2: 视觉处理

    Args:
        video_dir: 视频目录名
        output_dir: 输出目录
        audio_result: Stage 1 的输出 (可选)
        skip_captions: 跳过 caption 生成 (复用现有 visual.json 的 captions, 或置空)

    Returns:
        visual.json 数据 (dict)
    """
    config = get_config()
    from src.pipeline.orchestrator import resolve_video_path
    video_path = resolve_video_path(video_dir)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    print("=" * 60)
    print("Stage 2: 视觉处理")
    print(f"视频: {video_dir}")
    print("=" * 60)

    # ---- 路径选择: 有声纹 segments 走锚点驱动, 否则回退 SBD ----
    audio_segs = (audio_result or {}).get("segments") or []
    has_speaker = any((s.get("speaker_pred") or "").strip() for s in audio_segs)
    use_anchor = bool(audio_segs) and has_speaker

    if use_anchor:
        # ===== 新路径: 声纹锚点驱动 =====
        print(f"[Stage 2a] 声纹锚点生成 (segments={len(audio_segs)}) ...")
        content_range = (audio_result or {}).get("content_range") or {}
        anchors = build_anchors(
            audio_segs,
            content_range=content_range if content_range else None,
        )
        from collections import Counter
        type_ct = Counter(a.anchor_type for a in anchors)
        print(f"  生成 {len(anchors)} 个锚点: "
              f"midpoint={type_ct.get('midpoint', 0)}, "
              f"switch={type_ct.get('switch', 0)}, "
              f"silence={type_ct.get('silence', 0)}")

        print("[Stage 2b] 按锚点采样关键帧 ...")
        scenes = extract_anchor_keyframes(video_path, anchors, output_dir, video_id=video_dir)
        total_kf = sum(len(s.get("keyframe_paths", [])) for s in scenes)
        print(f"  写入 {total_kf} 张关键帧 (每锚点 1 帧)")
    else:
        # ===== 旧路径回退: SBD + 均匀采样 =====
        print(f"[Stage 2a] 场景检测 ({config.scene_detector}, 回退模式) ...")
        detector = create_detector(config)
        scene_objs = detector.detect_scenes(video_path)
        scenes = [s.to_dict() for s in scene_objs]
        print(f"  检测到 {len(scenes)} 个场景")

        print("[Stage 2b] 关键帧提取 (回退: 均匀采样) ...")
        scenes = extract_keyframes(
            video_path, scenes, output_dir,
            samples_per_scene=config.samples_per_scene,
        )
        total_kf = sum(len(s.get("keyframe_paths", [])) for s in scenes)
        print(f"  提取 {total_kf} 个关键帧")

        if hasattr(detector, "release"):
            detector.release()
            print("  检测器模型已释放")

        print("[Stage 2c] 帧过滤 (回退: 首中末) ...")
        scenes = filter_subtitle_frames(scenes, audio_result)

    # 保存场景数据
    save_json(scenes, os.path.join(output_dir, "scenes.json"))

    # Step 2d: OCR (多帧采样, 用 qwen3-vl-plus)
    print(f"[Stage 2d] 字幕 OCR (model={config.model_ocr}, 多帧采样)...")
    ocr_results = run_ocr(scenes, config, video_path=video_path)
    print(f"  OCR 完成: {len(ocr_results)} 个场景有结果")

    # Step 2e: 视觉描述
    captions = {}
    if skip_captions:
        # 复用现有 visual.json 的 captions (避免重跑, 用户已要求冻结)
        existing_visual = os.path.join(output_dir, "visual.json")
        if os.path.isfile(existing_visual):
            try:
                with open(existing_visual, "r", encoding="utf-8") as f:
                    old = json.load(f)
                captions = old.get("captions", {}) or {}
                print(f"[Stage 2e] caption 跳过 (复用现有 {len(captions)} 条)")
            except Exception:
                print("[Stage 2e] caption 跳过 (无法读取旧 visual.json, 留空)")
        else:
            print("[Stage 2e] caption 跳过 (无现有 visual.json, 留空)")
    else:
        print("[Stage 2e] 视觉描述生成...")
        captions = run_captions(scenes, audio_result, config)
        print(f"  Caption 完成: {len(captions)} 个场景有描述")

    # 合并输出
    visual_data = {
        "scenes": scenes,
        "ocr": ocr_results,
        "captions": captions,
    }
    save_json(visual_data, os.path.join(output_dir, "visual.json"))

    print(f"\n{'=' * 60}")
    print(f"结果: {os.path.join(output_dir, 'visual.json')}")
    print(f"场景数: {len(scenes)}, OCR: {len(ocr_results)}, Caption: {len(captions)}")
    print(f"{'=' * 60}")

    return visual_data
