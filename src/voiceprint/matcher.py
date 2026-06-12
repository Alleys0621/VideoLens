"""声纹识别匹配器 — 方法 C: 逐段识别

对每个 TranscriptSegment 独立进行 1:N 声纹识别，
直接替换 speaker_id 为声纹库中的真实角色名。
"""

import os
import shutil
import tempfile
from pathlib import Path

from src.core.logging import get_logger
from src.core.models.transcript import TranscriptSegment
from src.voiceprint.client import VoiceprintClient, cut_audio_segment

logger = get_logger()

# 讯飞声纹 API 建议音频时长 ≥ 3 秒
DEFAULT_MIN_DURATION = 3.0


def match_per_segment(
    segments: list[TranscriptSegment],
    audio_path: str,
    client: VoiceprintClient,
    *,
    score_threshold: float = 0.3,
    min_duration: float = DEFAULT_MIN_DURATION,
    name_mapping: dict[str, str] | None = None,
    api_delay: float = 0.1,
) -> tuple[list[TranscriptSegment], dict]:
    """逐段声纹识别（方法 C）。

    Args:
        segments: 带 speaker_id 的台词片段列表
        audio_path: 完整音频文件路径
        client: 讯飞声纹 API 客户端
        score_threshold: 匹配置信度阈值
        min_duration: 最短音频片段时长（秒），低于此跳过
        name_mapping: 声纹库 ID → 显示名称映射（如 {"xiyangyang": "喜羊羊"}）
        api_delay: API 调用间隔（秒），避免限频

    Returns:
        (segments, report): segments 的 speaker_id 被替换，report 为统计信息
    """
    import time

    name_mapping = name_mapping or {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="voiceprint_"))

    matched_count = 0
    skipped_count = 0
    failed_count = 0
    total_count = 0
    segment_details: list[dict] = []

    logger.info(f"声纹识别开始: {len(segments)} 个片段, 最短时长 {min_duration}s")

    try:
        for i, seg in enumerate(segments):
            duration = seg.end_time - seg.start_time

            # 跳过过短的片段
            if duration < min_duration:
                skipped_count += 1
                segment_details.append({
                    "index": i,
                    "status": "skipped",
                    "reason": f"时长 {duration:.1f}s < {min_duration}s",
                    "original_speaker": seg.speaker_id,
                })
                continue

            # 切音频
            wav_path = str(tmp_dir / f"seg_{i:04d}.wav")
            try:
                cut_audio_segment(audio_path, seg.start_time, seg.end_time, wav_path)
            except RuntimeError as e:
                skipped_count += 1
                segment_details.append({
                    "index": i,
                    "status": "skipped",
                    "reason": f"切片失败: {e}",
                    "original_speaker": seg.speaker_id,
                })
                continue

            # 检查文件大小
            fsize = os.path.getsize(wav_path) if os.path.isfile(wav_path) else 0
            if fsize < 1000:
                skipped_count += 1
                segment_details.append({
                    "index": i,
                    "status": "skipped",
                    "reason": f"文件过小 {fsize} bytes",
                    "original_speaker": seg.speaker_id,
                })
                continue

            total_count += 1

            # 1:N 声纹识别
            try:
                result = client.search(wav_path, top_k=1)
            except Exception as e:
                failed_count += 1
                segment_details.append({
                    "index": i,
                    "status": "error",
                    "reason": str(e),
                    "original_speaker": seg.speaker_id,
                })
                continue

            # 解析结果
            if result["code"] != 0 or "data" not in result:
                failed_count += 1
                segment_details.append({
                    "index": i,
                    "status": "api_error",
                    "reason": result.get("message", "unknown"),
                    "original_speaker": seg.speaker_id,
                })
                continue

            data = result["data"]
            score_list = data.get("scoreList", data.get("results", []))

            if not score_list:
                segment_details.append({
                    "index": i,
                    "status": "no_match",
                    "original_speaker": seg.speaker_id,
                })
                continue

            top = score_list[0]
            raw_id = top.get("featureId", "")
            score = top.get("score", 0)

            # 从 featureId 提取角色名（如 "xiyangyang_01" → "xiyangyang"）
            char_id = raw_id.rsplit("_", 1)[0] if "_" in raw_id else raw_id

            if score >= score_threshold:
                # 映射为显示名称
                display_name = name_mapping.get(char_id, char_id)
                seg.speaker_id = display_name
                matched_count += 1
                segment_details.append({
                    "index": i,
                    "status": "matched",
                    "feature_id": raw_id,
                    "character": display_name,
                    "score": score,
                    "original_speaker": seg.speaker_id if display_name == seg.speaker_id else seg.speaker_id,
                })
            else:
                segment_details.append({
                    "index": i,
                    "status": "below_threshold",
                    "feature_id": raw_id,
                    "character": name_mapping.get(char_id, char_id),
                    "score": score,
                    "original_speaker": seg.speaker_id,
                })

            # 避免限频
            if api_delay > 0:
                time.sleep(api_delay)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    report = {
        "total_segments": len(segments),
        "segments_sent": total_count,
        "matched": matched_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "match_rate": round(matched_count / total_count, 3) if total_count > 0 else 0,
        "segment_details": segment_details,
    }

    logger.info(
        f"声纹识别完成: {matched_count}/{total_count} 匹配 "
        f"({skipped_count} 跳过, {failed_count} 失败)"
    )

    return segments, report
