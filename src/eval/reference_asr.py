"""
omni-plus 参考 ASR — 评估专用, 不进生产 pipeline.

设计动机:
  Stage 1 用 omni-flash 做 ASR (快/便宜), 但其核心目的是说话人轮次切分,
  ASR 文本本身有错误 (~5-10%), 不能用作字幕 OCR 的准确率参考.
  本模块用 omni-plus 重跑一遍全集 ASR, 牺牲成本换准确率,
  作为 OCR precision/recall 的软 GT.

输出: data/output/{video}/reference_asr.json
  {
    "video": "...",
    "model": "qwen3.5-omni-plus",
    "segments": [{"start_time": float, "end_time": float, "text": str}, ...],
    "usage": {...}
  }

成本 (22 min 单集, omni-plus):
  ~22 chunks × ~1.5k audio_tokens + ~1k text_out
  ≈ 33k audio_in × ¥53/M + 22k text_out × ¥213/M
  ≈ ¥1.75 + ¥4.7 ≈ ¥6.4 / 集
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from openai import OpenAI

from src.core.config import get_config
from src.core.cost import get_cost_tracker
from src.core.llm.base_client import _report_usage
from src.core.logging import get_logger
from src.core.path_utils import resolve_video_path

logger = get_logger()

# 参考 ASR 专用 prompt — 比 Stage 1 简化, 聚焦精准文本
REFERENCE_ASR_PROMPT = """请精准转录这段音频中所有的人声对话内容。

要求:
1. 按自然语义句为单位切分 (一个完整句子一段, 不按说话人轮次)
2. 每段输出 start_time / end_time (秒, 精确到 0.001; 当前 chunk 内部时间轴, 起点 = 0)
3. 严格逐字转录原文, 不修改、不补全、不加标点以外字符
4. 跳过纯音乐段、纯音效段、笑声/哭声等非语义内容
5. 一句话不超过 30 字, 长句按自然停顿拆分
6. 只输出 JSON, 不要 Markdown / 不要解释

输出格式:
{
  "segments": [
    {"start_time": 0.000, "end_time": 1.234, "text": "你好"},
    {"start_time": 1.500, "end_time": 3.200, "text": "今天天气不错"}
  ]
}
"""

CHUNK_DURATION = 60.0  # 秒, 与 Stage 1 一致


@dataclass
class ReferenceSegment:
    start_time: float
    end_time: float
    text: str


def _extract_audio_if_needed(video_path: str, output_dir: str) -> str:
    """若 audio.wav 已存在 (Stage 1 跑过) 则复用, 否则从视频新抽."""
    audio_path = os.path.join(output_dir, "audio.wav")
    if os.path.isfile(audio_path) and os.path.getsize(audio_path) > 1000:
        return audio_path
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", audio_path,
    ], capture_output=True, timeout=300, check=True)
    return audio_path


def _get_audio_duration(audio_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-i", audio_path, "-show_entries", "format=duration",
         "-v", "quiet", "-of", "csv=p=0"],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip()) if r.stdout.strip() else 0.0


def _cut_chunk(audio_path: str, start: float, dur: float) -> bytes:
    """切一个 chunk, 返回 wav bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(dur),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name,
        ], capture_output=True, timeout=30, check=True)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try: os.remove(tmp.name)
        except OSError: pass


def _call_omni_plus(client: OpenAI, model: str, wav_bytes: bytes, chunk_dur: float):
    """单 chunk 调用 omni-plus ASR."""
    b64 = base64.b64encode(wav_bytes).decode()
    prompt = REFERENCE_ASR_PROMPT.replace(
        "当前 chunk", f"当前 chunk (时长 {chunk_dur:.1f}s)"
    )
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
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
    usage = None
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            full_text += chunk.choices[0].delta.content
        if getattr(chunk, "usage", None):
            usage = chunk.usage

    latency = time.time() - t0
    if usage is not None:
        _report_usage(model, usage, latency, "eval_reference_asr")
    return full_text, usage, latency


def _parse_response(text: str, chunk_start: float) -> list[dict]:
    """从 LLM 响应中抽出 segments, 把 chunk 内时间戳映射到全集时间."""
    from src.core.helpers.text_utils import extract_json_obj
    parsed = extract_json_obj(text)
    if not parsed or not isinstance(parsed, dict):
        return []
    out = []
    for seg in parsed.get("segments", []):
        try:
            ls = float(seg.get("start_time", 0))
            le = float(seg.get("end_time", 0))
            txt = (seg.get("text", "") or "").strip()
        except (TypeError, ValueError):
            continue
        if le <= ls or not txt:
            continue
        out.append({
            "start_time": round(chunk_start + ls, 3),
            "end_time": round(chunk_start + le, 3),
            "text": txt,
        })
    return out


def run_reference_asr(
    video_dir: str,
    output_dir: str,
    video_path: str | None = None,
    chunk_duration: float = CHUNK_DURATION,
    max_workers: int = 4,
) -> dict:
    """对一集视频跑 omni-plus 参考 ASR.

    Args:
        video_dir: 视频目录名 (如 "家有儿女/第一季/第01集")
        output_dir: 输出目录
        video_path: 视频路径; None 则用 resolve_video_path
        chunk_duration: chunk 时长 (秒)
        max_workers: chunk 并发数

    Returns:
        reference_asr 数据 dict, 同时写入 {output_dir}/reference_asr.json
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config = get_config()
    model = config.model_omni_plus
    api_key = config.dashscope_api_key
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    if video_path is None:
        video_path = resolve_video_path(video_dir)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    audio_path = _extract_audio_if_needed(video_path, output_dir)
    duration = _get_audio_duration(audio_path)
    print(f"[reference_asr] {video_dir}  audio={duration:.1f}s  model={model}")

    # 切 chunks
    chunks: list[tuple[float, float]] = []
    pos = 0.0
    while pos < duration:
        end = min(pos + chunk_duration, duration)
        chunks.append((pos, end))
        pos = end
    print(f"  chunks: {len(chunks)} × {chunk_duration:.0f}s")

    # 并发跑 ASR
    os.makedirs(output_dir, exist_ok=True)

    def _process_chunk(idx_and_range):
        idx, (start, end) = idx_and_range
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            wav = _cut_chunk(audio_path, start, end - start)
            if len(wav) < 1000: return idx, start, []
            text, _, _ = _call_omni_plus(client, model, wav, end - start)
            return idx, start, _parse_response(text, start)
        except Exception as e:
            logger.warning(f"reference_asr chunk {idx} 失败: {e}")
            return idx, start, []

    all_segments: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_process_chunk, (i, c)) for i, c in enumerate(chunks)]
        completed = 0
        for fut in as_completed(futs):
            _, _, segs = fut.result()
            all_segments.extend(segs)
            completed += 1
            if completed % 5 == 0 or completed == len(chunks):
                print(f"  进度: {completed}/{len(chunks)}, 累计 {len(all_segments)} segments")

    # 按时间排序
    all_segments.sort(key=lambda s: s["start_time"])

    data = {
        "video": video_dir,
        "model": model,
        "audio_duration": round(duration, 2),
        "chunk_duration": chunk_duration,
        "segments": all_segments,
    }
    out_path = os.path.join(output_dir, "reference_asr.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  写入 {out_path}: {len(all_segments)} segments")

    return data
