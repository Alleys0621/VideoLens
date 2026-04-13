"""测试说话人识别 + 对齐模块"""

import os
import json
from pathlib import Path

# 加载 .env
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.is_file():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from vl.core.models.transcript import TranscriptSegment, DiarizationSegment
from vl.asr.diarizer import Diarizer
from vl.asr.aligner import Aligner

AUDIO_PATH = "data/output/preprocessing/052 鸟蛋之争.wav"
TRANSCRIPT_PATH = "data/output/transcripts/052 鸟蛋之争/transcript.json"


def test_diarizer():
    """测试说话人识别"""
    print("=" * 60)
    print("[Test 1] 说话人识别 (Diarizer)")
    print("=" * 60)

    diarizer = Diarizer()
    results = diarizer.diarize(AUDIO_PATH)

    assert len(results) > 0, "应该识别到说话人片段"
    print(f"  识别到 {len(results)} 个说话人片段")

    speakers = set(r.speaker for r in results)
    print(f"  识别到 {len(speakers)} 个说话人: {sorted(speakers)}")

    # 每个说话人的总时长
    speaker_duration = {}
    for r in results:
        dur = r.end_time - r.start_time
        speaker_duration[r.speaker] = speaker_duration.get(r.speaker, 0) + dur
    print("\n  各说话人时长:")
    for spk in sorted(speaker_duration, key=lambda x: -speaker_duration[x]):
        print(f"    {spk}: {speaker_duration[spk]:.1f}s")

    # 前 5 个片段
    print("\n  前 5 个片段:")
    for r in results[:5]:
        print(f"    {r.speaker}: {r.start_time:.1f}s - {r.end_time:.1f}s ({r.end_time - r.start_time:.1f}s)")

    print("\n  [PASS] 说话人识别测试通过")
    return results


def test_aligner(diarization_results):
    """测试说话人对齐"""
    print("\n" + "=" * 60)
    print("[Test 2] 说话人对齐 (Aligner)")
    print("=" * 60)

    # 加载已有的转录结果
    raw = json.load(open(TRANSCRIPT_PATH, "r", encoding="utf-8"))
    segments = [TranscriptSegment.from_dict(s) for s in raw]
    print(f"  加载 {len(segments)} 个转录片段")

    # 对齐前: 全部是 SPEAKER_00
    before_speakers = set(s.speaker_id for s in segments)
    print(f"  对齐前说话人: {before_speakers}")

    aligner = Aligner()
    aligned = aligner.align(segments, diarization_results)

    # 对齐后
    after_speakers = set(s.speaker_id for s in aligned)
    print(f"  对齐后说话人: {after_speakers}")

    # 统计每个说话人的片段数
    speaker_count = {}
    for s in aligned:
        speaker_count[s.speaker_id] = speaker_count.get(s.speaker_id, 0) + 1
    print("\n  各说话人片段数:")
    for spk in sorted(speaker_count, key=lambda x: -speaker_count[x]):
        print(f"    {spk}: {speaker_count[spk]} 个片段")

    # 展示几条对齐结果
    print("\n  示例对齐结果:")
    for s in aligned[:8]:
        text = s.text[:40] if len(s.text) > 40 else s.text
        print(f"    [{s.speaker_id}] {s.start_time:.1f}s-{s.end_time:.1f}s: {text}")

    assert len(after_speakers) > 1, "对齐后应该有多个说话人"
    print("\n  [PASS] 说话人对齐测试通过")


if __name__ == "__main__":
    results = test_diarizer()
    test_aligner(results)
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
