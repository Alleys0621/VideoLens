"""
Stage 1: 音频处理

流程:
  1. 提取音频 (ffmpeg)
  2. 片头/片尾曲检测 (qwen3.5-omni-plus)
  3. 截取正文音频
  4. 预处理: 静音检测 + 智能切块
  5. Omni Flash 识别 (ASR + 说话人轮次 + 情感 + 时间戳)
  6. 映射回原始时间轴
  7. 讯飞声纹 1:N 识别
  → audio.json
"""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass

from openai import OpenAI

from src.core.config import get_config
from src.core.cost import get_cost_tracker
from src.core.helpers.text_utils import extract_json_obj
from src.core.llm.base_client import _report_usage
from src.core.logging import get_logger

logger = get_logger()
from src.core.path_utils import resolve_video_path

if not hasattr(sys.stdout, 'reconfigure'):
    pass

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

THEME_MODEL = "qwen3.5-omni-plus"
OMNI_MODEL = "qwen3.5-omni-flash"
THEME_WINDOW = 120
CHUNK_DURATION = 60
SILENCE_DB = -30
SILENCE_REMOVE_MIN = 0.3
SPEECH_REMOVE_MAX = 0.3

NAME_MAP = {}  # 由 pipeline.yaml voiceprint_groups 动态注入


# ══════════════════════════════════════════════════════════════
# 音频提取
# ══════════════════════════════════════════════════════════════

def extract_audio(video_path, output_path):
    """从视频中提取音频为 16kHz/16bit/mono WAV"""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 提取音频失败: {r.stderr[-300:]}")

    dur = float(subprocess.run(
        ["ffprobe", "-i", output_path, "-show_entries", "format=duration",
         "-v", "quiet", "-of", "csv=p=0"],
        capture_output=True, text=True,
    ).stdout.strip())
    return dur


# ══════════════════════════════════════════════════════════════
# 片头/片尾曲检测
# ══════════════════════════════════════════════════════════════

def detect_theme_songs(client, audio_path, total_duration, prompt_text):
    """检测片头曲和片尾曲, 返回 (content_start, content_end, theme_info)"""
    print("[Stage 1b] 片头/片尾曲检测...")

    theme_info = {"opening": None, "ending": None}
    content_start = 0.0
    content_end = total_duration

    if total_duration <= THEME_WINDOW * 2:
        print(f"  音频过短 ({total_duration:.1f}s), 跳过主题曲检测")
        return content_start, content_end, theme_info

    for window_type, offset in [("opening", 0), ("ending", max(0, total_duration - THEME_WINDOW))]:
        clip_dur = min(THEME_WINDOW, total_duration - offset)
        if clip_dur < 10:
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(offset), "-t", str(clip_dur),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name,
            ], capture_output=True, timeout=30)
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"  {window_type} 片段截取失败: {e}")
            continue
        finally:
            if os.path.isfile(tmp.name):
                os.remove(tmp.name)

        try:
            t_call = time.time()
            response = client.chat.completions.create(
                model=THEME_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {
                            "data": f"data:audio/wav;base64,{b64}", "format": "wav",
                        }},
                        {"type": "text", "text": prompt_text},
                    ],
                }],
                modalities=["text"],
                stream=True,
                stream_options={"include_usage": True},
            )

            full_text = ""
            final_usage = None
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                if getattr(chunk, "usage", None):
                    final_usage = chunk.usage

            if final_usage is not None:
                _report_usage(THEME_MODEL, final_usage, time.time() - t_call, "stage1_theme")

            raw_clean = full_text.strip()
            if raw_clean.startswith("```"):
                raw_clean = raw_clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = None
            try:
                parsed = json.loads(raw_clean)
            except json.JSONDecodeError:
                parsed = extract_json_obj(full_text)
            if not parsed:
                print(f"  {window_type}: 解析失败")
                continue

            ts = parsed.get("theme_song", parsed)
            if not ts.get("exist"):
                print(f"  {window_type}: 未检测到主题曲")
                continue

            local_start = float(ts.get("start_time", 0))
            local_end = float(ts.get("end_time", 0))
            orig_start = offset + local_start
            orig_end = offset + local_end
            theme_type = ts.get("type", window_type)

            theme_info[window_type] = {
                "type": theme_type,
                "start_time": round(orig_start, 1),
                "end_time": round(orig_end, 1),
                "duration": round(orig_end - orig_start, 1),
            }
            print(f"  {window_type}: {theme_type} [{orig_start:.1f}s - {orig_end:.1f}s] ({orig_end - orig_start:.1f}s)")

        except Exception as e:
            print(f"  {window_type} 检测失败: {e}")

    if theme_info["opening"]:
        content_start = theme_info["opening"]["end_time"]
    if theme_info["ending"]:
        content_end = theme_info["ending"]["start_time"]

    if content_start >= content_end:
        print(f"  主题曲范围异常 ({content_start:.1f} >= {content_end:.1f}), 使用完整音频")
        content_start = 0.0
        content_end = total_duration
        theme_info = {"opening": None, "ending": None}

    content_dur = content_end - content_start
    print(f"  正文范围: [{content_start:.1f}s - {content_end:.1f}s] ({content_dur:.1f}s)")
    return content_start, content_end, theme_info


def extract_content_audio(audio_path, content_start, content_end, output_path):
    """从原始音频中截取正文部分"""
    dur = content_end - content_start
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(content_start), "-t", str(dur),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 截取正文音频失败: {r.stderr[-300:]}")
    return dur


# ══════════════════════════════════════════════════════════════
# 音频预处理
# ══════════════════════════════════════════════════════════════

@dataclass
class Segment:
    orig_start: float
    orig_end: float
    new_start: float = 0.0
    new_end: float = 0.0


def preprocess_audio(audio_path, output_dir):
    """静默检测, 生成预处理音频 + 时间映射 + 候选切割点"""
    print("[Stage 1c] 音频预处理...")

    r = subprocess.run([
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_REMOVE_MIN}",
        "-f", "null", "-",
    ], capture_output=True, timeout=120)

    silences = []
    stderr = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
    for line in stderr.split("\n"):
        if "silence_start" in line:
            t = float(line.split("silence_start:")[1].strip().split()[0])
            silences.append({"start": t})
        elif "silence_end" in line:
            t = float(line.split("silence_end:")[1].strip().split("|")[0].strip())
            if silences and "end" not in silences[-1]:
                silences[-1]["end"] = t

    total_dur = float(subprocess.run(
        ["ffprobe", "-i", audio_path, "-show_entries", "format=duration",
         "-v", "quiet", "-of", "csv=p=0"],
        capture_output=True, text=True,
    ).stdout.strip())

    if silences and "end" not in silences[-1]:
        silences[-1]["end"] = total_dur

    speech_segs = []
    prev_end = 0.0
    for sil in silences:
        if prev_end < sil["start"]:
            speech_segs.append((prev_end, sil["start"]))
        prev_end = sil["end"]
    if prev_end < total_dur:
        speech_segs.append((prev_end, total_dur))
    if not silences:
        speech_segs = [(0.0, total_dur)]

    filtered = [(s, e) for s, e in speech_segs if (e - s) >= SPEECH_REMOVE_MAX]
    filtered_dur = sum(e - s for s, e in filtered)
    print(f"  总时长: {total_dur:.1f}s, 静音段: {len(silences)}, 有声段: {len(filtered)}, 预处理后: {filtered_dur:.1f}s")

    segments = []
    new_t = 0.0
    for s, e in filtered:
        seg = Segment(orig_start=s, orig_end=e, new_start=new_t, new_end=new_t + (e - s))
        segments.append(seg)
        new_t += e - s

    candidate_cuts = [segments[i].new_start for i in range(1, len(segments))]

    pp_path = os.path.join(output_dir, "audio_preprocessed.wav")
    tmp_dir = tempfile.mkdtemp(prefix="pp_seg_")
    seg_paths = []

    for i, seg in enumerate(segments):
        sp = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(seg.orig_start), "-to", str(seg.orig_end),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", sp,
        ], capture_output=True, timeout=30)
        if os.path.isfile(sp) and os.path.getsize(sp) > 44:
            seg_paths.append(sp)

    list_file = os.path.join(tmp_dir, "list.txt")
    with open(list_file, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", pp_path,
    ], capture_output=True, timeout=120)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    mapping = [
        {"orig_start": round(s.orig_start, 3), "orig_end": round(s.orig_end, 3),
         "new_start": round(s.new_start, 3), "new_end": round(s.new_end, 3)}
        for s in segments
    ]
    with open(os.path.join(output_dir, "timeline_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return pp_path, segments, candidate_cuts, filtered_dur


# ══════════════════════════════════════════════════════════════
# 智能切块
# ══════════════════════════════════════════════════════════════

def smart_chunk(pp_duration, target_dur, candidate_cuts):
    chunks = []
    pos = 0.0
    while pos < pp_duration:
        target_end = pos + target_dur
        if target_end >= pp_duration:
            chunks.append((pos, pp_duration))
            break
        best_cut, best_dist = None, float("inf")
        for c in candidate_cuts:
            if c <= pos or c >= pp_duration:
                continue
            dist = abs(c - target_end)
            if dist < best_dist:
                best_dist, best_cut = dist, c
        if best_cut and best_dist < target_dur * 0.3 and best_cut > pos + 5:
            chunks.append((pos, best_cut))
            pos = best_cut
        else:
            chunks.append((pos, target_end))
            pos = target_end
    return chunks


# ══════════════════════════════════════════════════════════════
# 时间映射
# ══════════════════════════════════════════════════════════════

def map_to_original(local_time, chunk_start_new, segments):
    abs_new = chunk_start_new + local_time
    for seg in segments:
        if seg.new_start <= abs_new <= seg.new_end:
            ratio = (abs_new - seg.new_start) / max(seg.new_end - seg.new_start, 0.001)
            return seg.orig_start + ratio * (seg.orig_end - seg.orig_start)
    if abs_new <= segments[0].new_start:
        return segments[0].orig_start
    if abs_new >= segments[-1].new_end:
        return segments[-1].orig_end
    return abs_new


# ══════════════════════════════════════════════════════════════
# Omni 识别
# ══════════════════════════════════════════════════════════════

def omni_recognize(client, pp_path, segments, chunks, user_template):
    print("[Stage 1d] Omni 识别...")
    all_dialogues = []
    total_in = total_out = 0
    total_latency = 0
    total_cost = 0.0

    tracker = get_cost_tracker()

    for ci, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_dur = chunk_end - chunk_start
        if chunk_dur < 1.0:
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", pp_path,
                "-ss", str(chunk_start), "-t", str(chunk_dur),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name,
            ], capture_output=True, timeout=30)
            if not os.path.isfile(tmp.name) or os.path.getsize(tmp.name) < 1000:
                continue
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.warning(f"ffmpeg chunk 切分失败, 跳过: {e}")
            continue
        finally:
            if os.path.isfile(tmp.name):
                os.remove(tmp.name)

        prompt = user_template.format(chunk_start_time=0, chunk_end_time=round(chunk_dur, 1))

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=OMNI_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {
                            "data": f"data:audio/wav;base64,{b64}", "format": "wav",
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
                modalities=["text"],
                stream=True,
                stream_options={"include_usage": True},
            )

            full_text = ""
            chunk_usage = None
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                if getattr(chunk, "usage", None):
                    chunk_usage = chunk.usage

            # 上报分模态成本 (CostTracker 内部按 omni 定价表自动分档)
            chunk_latency = time.time() - t0
            prev_cost = tracker.total_cost
            if chunk_usage is not None:
                _report_usage(OMNI_MODEL, chunk_usage, chunk_latency, "stage1_dialogue")
                total_in += getattr(chunk_usage, "prompt_tokens", 0) or 0
                total_out += getattr(chunk_usage, "completion_tokens", 0) or 0
            total_cost += tracker.total_cost - prev_cost
            total_latency += chunk_latency

            parsed = extract_json_obj(full_text)
            if parsed and isinstance(parsed, dict):
                for d in parsed.get("dialogues", []):
                    ls = float(d.get("start_time", 0))
                    le = float(d.get("end_time", 0))
                    orig_s = map_to_original(ls, chunk_start, segments)
                    orig_e = map_to_original(le, chunk_start, segments)
                    all_dialogues.append({
                        "start_time": round(orig_s, 3),
                        "end_time": round(orig_e, 3),
                        "text": d.get("text", ""),
                        "emotion": d.get("emotion", ""),
                    })

            if (ci + 1) % 10 == 0:
                print(f"  chunk {ci+1}/{len(chunks)} done, {len(all_dialogues)} segments")

        except Exception as e:
            print(f"  chunk {ci} fail: {e}")

    print(f"  完成: {len(all_dialogues)} 段, ¥{total_cost:.4f}, {total_latency:.0f}s")
    return all_dialogues, {"input_tokens": total_in, "output_tokens": total_out, "cost": total_cost}


# ══════════════════════════════════════════════════════════════
# 讯飞声纹
# ══════════════════════════════════════════════════════════════

def run_iflytek(dialogues, audio_path, app_id, api_key, api_secret, group_id, name_map=None):
    from src.voiceprint.client import VoiceprintClient, cut_audio_segment

    _name_map = name_map or NAME_MAP
    print(f"[Stage 1e] 讯飞声纹识别... (group={group_id}, {len(_name_map)} 角色)")
    client = VoiceprintClient(app_id, api_key, api_secret, group_id, verbose=False)
    tmp_dir = tempfile.mkdtemp(prefix="spk_pipe_")

    results = []
    for idx, seg in enumerate(dialogues):
        start, end = seg["start_time"], seg["end_time"]
        text = seg.get("text", "").strip()
        dur = end - start

        if dur < 0.5 or not text or text in ("(歌曲)", "(笑声)", "(音效)"):
            results.append({"pred": "", "score": 0.0})
            continue

        clip = os.path.join(tmp_dir, f"s_{idx:04d}.wav")
        try:
            cut_audio_segment(audio_path, start, end, clip)
            raw = client.search(clip, top_k=1)
            data = raw.get("data", {})
            sl = data.get("scoreList", [])
            if sl:
                raw_id = sl[0].get("featureId", "")
                score = sl[0].get("score", 0)
                cid = raw_id.rsplit("_", 1)[0] if "_" in raw_id else raw_id
                pred = _name_map.get(cid, cid)
            else:
                pred, score = "", 0
            results.append({"pred": pred, "score": round(score, 4)})
        except Exception as e:
            logger.warning(f"ASR chunk 预测失败, 用空兜底: {e}")
            results.append({"pred": "", "score": 0.0})

        time.sleep(0.1)
        if (idx + 1) % 20 == 0:
            identified = sum(1 for r in results if r["pred"])
            print(f"  {idx+1}/{len(dialogues)} done, {identified} identified")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    identified = sum(1 for r in results if r["pred"])
    print(f"  完成: {identified}/{len(dialogues)} 段有声纹预测")
    return results


# ══════════════════════════════════════════════════════════════
# 输出生成
# ══════════════════════════════════════════════════════════════

def generate_output(dialogues, iflytek_results, vp_threshold=0.4):
    """生成 speaker_pred 格式的段列表

    Args:
        dialogues: Omni 识别结果
        iflytek_results: 声纹识别结果
        vp_threshold: 声纹置信度阈值，低于此值统一标记为 '路人' (默认 0.4)
    """
    output = []
    for i, seg in enumerate(dialogues):
        vp = iflytek_results[i] if i < len(iflytek_results) else {"pred": "", "score": 0}
        text = seg.get("text", "").strip()
        if not text:
            continue

        raw = f"{seg['start_time']:.3f}_{seg['end_time']:.3f}_{text[:20]}"
        seg_id = hashlib.md5(raw.encode()).hexdigest()[:8]

        pred = vp["pred"]
        score = vp["score"]
        # 低于阈值一律标路人 (含 no_match 的 pred="" 情况)
        if vp_threshold > 0 and score < vp_threshold:
            pred = "路人"

        output.append({
            "segment_id": seg_id,
            "begin_time": seg["start_time"],
            "end_time": seg["end_time"],
            "speaker_pred": pred,
            "vp_score": score,
            "text": text,
            "emotion": seg.get("emotion", ""),
            "speaker_gt": "",
            "emotion_gt": "",
        })

    return output


# ══════════════════════════════════════════════════════════════
# Stage 1 入口
# ══════════════════════════════════════════════════════════════

def run_stage1(video_dir: str, output_dir: str, skip_theme: bool = False, chunk_dur: int = 60, vp_threshold: float = 0.4, group_id: str = "", name_map: dict = None) -> dict:
    """Stage 1: 音频处理

    Args:
        video_dir: 视频目录名, 如 "052 鸟蛋之争"
        output_dir: 输出目录
        skip_theme: 是否跳过片头/片尾曲检测
        chunk_dur: Omni chunk 时长 (秒)
        vp_threshold: 声纹置信度阈值, 低于此值统一标记为 '路人' (默认 0.4, 0=不过滤)
        group_id: 讯飞声纹组 ID (空则跳过声纹识别)
        name_map: 声纹 featureId → 角色名映射

    Returns:
        audio.json 数据 (dict)
    """
    import yaml
    from dotenv import load_dotenv
    load_dotenv()

    # 路径
    video_path = resolve_video_path(video_dir)
    audio_path = os.path.join(output_dir, "audio.wav")
    result_path = os.path.join(output_dir, "audio.json")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Stage 1: 音频处理")
    print(f"视频: {video_dir}")
    print(f"模型: {OMNI_MODEL}, Chunk: {chunk_dur}s")
    print("=" * 60)

    # Step 1: 提取音频
    if not os.path.isfile(audio_path):
        print("[Stage 1a] 提取音频...")
        dur = extract_audio(video_path, audio_path)
        print(f"  音频: {audio_path} ({dur:.1f}s)")
    else:
        dur = float(subprocess.run(
            ["ffprobe", "-i", audio_path, "-show_entries", "format=duration",
             "-v", "quiet", "-of", "csv=p=0"],
            capture_output=True, text=True,
        ).stdout.strip())
        print(f"[Stage 1a] 音频已存在: {audio_path} ({dur:.1f}s)")

    # 初始化 Omni 客户端
    _cfg = get_config()
    client = OpenAI(
        api_key=_cfg.dashscope_api_key,
        base_url=_cfg.dashscope_base_url,
    )

    # Step 1b: 片头/片尾曲检测
    content_start = 0.0
    content_end = dur
    theme_info = {"opening": None, "ending": None}

    if not skip_theme:
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        theme_prompt = prompts.get("stage1a_theme_detection", {}).get("user", "")
        content_start, content_end, theme_info = detect_theme_songs(client, audio_path, dur, theme_prompt)

    # 截取正文音频
    if content_start > 0 or content_end < dur:
        content_audio_path = os.path.join(output_dir, "audio_content.wav")
        if not os.path.isfile(content_audio_path):
            print(f"  截取正文音频: [{content_start:.1f}s - {content_end:.1f}s]")
            extract_content_audio(audio_path, content_start, content_end, content_audio_path)
        pipeline_audio = content_audio_path
    else:
        pipeline_audio = audio_path

    # Step 1c: 预处理
    pp_path, segments, candidate_cuts, pp_duration = preprocess_audio(pipeline_audio, output_dir)

    # 切块
    chunks = smart_chunk(pp_duration, chunk_dur, candidate_cuts)
    print(f"  {len(chunks)} 个 chunk (目标 {chunk_dur}s)")

    # Step 1d: Omni
    with open("config/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    omni_cfg = prompts.get("stage1b_dialogue_recognition", {})
    user_template = omni_cfg.get("user", "")
    dialogues, usage = omni_recognize(client, pp_path, segments, chunks, user_template)

    # 映射回原始时间轴
    if content_start > 0:
        print(f"  映射时间轴: +{content_start:.1f}s offset")
        for d in dialogues:
            d["start_time"] = round(d["start_time"] + content_start, 3)
            d["end_time"] = round(d["end_time"] + content_start, 3)

    # Step 1e: 讯飞
    if group_id:
        iflytek_results = run_iflytek(
            dialogues, audio_path,
            _cfg.xfyun_app_id,
            _cfg.xfyun_api_key,
            _cfg.xfyun_api_secret,
            group_id, name_map,
        )
    else:
        print("[Stage 1e] 未配置声纹库，跳过声纹识别")
        iflytek_results = [{"pred": "", "score": 0.0}] * len(dialogues)

    # 输出
    segments_out = generate_output(dialogues, iflytek_results, vp_threshold)

    result_data = {
        "theme_songs": theme_info,
        "content_range": {
            "start": round(content_start, 1),
            "end": round(content_end, 1),
        },
        "segments": segments_out,
        "usage": usage,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 统计
    with_pred = sum(1 for o in segments_out if o["speaker_pred"])
    by_spk = Counter(o["speaker_pred"] for o in segments_out if o["speaker_pred"])

    print(f"\n{'=' * 60}")
    print(f"结果: {result_path}")
    print(f"总段数: {len(segments_out)}, 有声纹预测: {with_pred}")
    if theme_info["opening"] or theme_info["ending"]:
        print(f"主题曲: {theme_info}")
    print(f"角色分布:")
    for spk, cnt in by_spk.most_common():
        print(f"  {spk}: {cnt} 段")
    print(f"费用: ¥{usage['cost']:.4f}")
    print(f"{'=' * 60}")

    return result_data
