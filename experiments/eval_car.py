"""
Character Attribution Rate (CAR) 评估脚本

定义:
  以 GT 台词段为评测单元。
  对于每个 GT 台词段，统计与其时间重叠的所有 Omni 输出片段中各角色的
  文本贡献量（重叠部分的文本长度），取贡献量最大的角色作为预测角色。
  若预测角色与 GT 角色一致，则记为正确。
  CAR = 正确 GT 台词段数 / GT 总台词段数

数据来源:
  - experiment 8 (exp8_iflytek_on_omni.json): 每段 Omni 的讯飞声纹预测结果
  - speaker_gt.json: GT 台词段
"""

import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT = Path(__file__).resolve().parent.parent
GT_PATH = PROJECT / "speaker_gt.json"
EXP8_PATH = PROJECT / "tests" / "experiment_results" / "exp8_iflytek_on_omni.json"
EXP9_VP_DIR = PROJECT / "tests" / "experiment_results"
OMNI_DIR = PROJECT / "tests" / "test_omni_result"

NO_VP = {"歌手", "鳄鱼妈妈"}


def load_gt():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        return [g for g in json.load(f) if g["speaker_gt"] not in NO_VP]


def overlap_duration(s1, e1, s2, e2):
    return max(0.0, min(e1, e2) - max(s1, s2))


def compute_car(gt_segments, omni_segments_with_speaker):
    """
    计算 CAR。

    Args:
        gt_segments: [{begin_time, end_time, speaker_gt, text}, ...]
        omni_segments_with_speaker: [{start, end, pred, text, dur}, ...]
            pred = 讯飞声纹预测角色 (空字符串=未识别)

    Returns:
        (car, details)
        details: [{gt_idx, gt_speaker, contributions, predicted, correct}, ...]
    """
    details = []

    for gi, g in enumerate(gt_segments):
        gs, ge = g["begin_time"], g["end_time"]
        gt_spk = g["speaker_gt"]

        # 收集所有重叠 Omni 段的贡献
        contributions = defaultdict(float)  # speaker -> total text contribution

        for seg in omni_segments_with_speaker:
            if not seg.get("pred"):
                continue
            ov = overlap_duration(gs, ge, seg["start"], seg["end"])
            if ov <= 0:
                continue

            # 文本贡献量 = 重叠比例 × 该段文本长度
            seg_dur = seg["end"] - seg["start"]
            if seg_dur <= 0:
                continue
            overlap_ratio = ov / seg_dur
            text_contrib = overlap_ratio * len(seg.get("text", ""))
            contributions[seg["pred"]] += text_contrib

        if not contributions:
            # 没有任何 Omni 段重叠 → 无法预测
            details.append({
                "gt_idx": gi,
                "gt_speaker": gt_spk,
                "gt_text": g["text"][:40],
                "contributions": {},
                "predicted": "",
                "correct": False,
                "reason": "no_overlap",
            })
            continue

        predicted = max(contributions, key=contributions.get)
        correct = predicted == gt_spk

        details.append({
            "gt_idx": gi,
            "gt_speaker": gt_spk,
            "gt_text": g["text"][:40],
            "contributions": {k: round(v, 2) for k, v in sorted(
                contributions.items(), key=lambda x: -x[1]
            )},
            "predicted": predicted,
            "correct": correct,
        })

    correct_count = sum(1 for d in details if d["correct"])
    car = correct_count / len(details) if details else 0.0

    return car, details


def compute_car_for_config(config_name, gt_segments, exp8_data):
    """从 experiment 8 数据计算某个配置的 CAR"""
    config_data = exp8_data.get(config_name)
    if not config_data:
        return None, None

    # 提取有讯飞预测的 Omni 段
    omni_segs = []
    for d in config_data["details"]:
        if d["status"] == "ok" and d.get("pred"):
            omni_segs.append({
                "start": d["start"],
                "end": d["end"],
                "pred": d["pred"],
                "text": d.get("text", ""),
                "score": d.get("score", 0),
            })

    car, details = compute_car(gt_segments, omni_segs)
    return car, details


def analyze_details(details, gt_segments):
    """分析 CAR 的细粒度指标"""
    valid = [d for d in details if d.get("reason") != "no_overlap"]

    # 按说话人统计
    by_speaker = defaultdict(lambda: {"total": 0, "correct": 0})
    for d in valid:
        spk = d["gt_speaker"]
        by_speaker[spk]["total"] += 1
        if d["correct"]:
            by_speaker[spk]["correct"] += 1

    # 按 GT 段时长统计
    by_duration = defaultdict(lambda: {"total": 0, "correct": 0})
    for d in valid:
        gi = d["gt_idx"]
        dur = gt_segments[gi]["end_time"] - gt_segments[gi]["begin_time"]
        if dur < 2:
            label = "<2s"
        elif dur < 5:
            label = "2-5s"
        elif dur < 10:
            label = "5-10s"
        else:
            label = ">10s"
        by_duration[label]["total"] += 1
        if d["correct"]:
            by_duration[label]["correct"] += 1

    # 按贡献集中度统计 (top1 贡献占比)
    concentration = []
    for d in valid:
        if d["contributions"]:
            vals = list(d["contributions"].values())
            total = sum(vals)
            top1_ratio = max(vals) / total if total > 0 else 0
            concentration.append(top1_ratio)

    import statistics
    avg_concentration = statistics.mean(concentration) if concentration else 0

    return by_speaker, by_duration, avg_concentration


def run_iflytek_on_omni(dialogues, gt_segments, audio_path):
    """对 Omni 对话结果跑讯飞声纹, 返回 [{start, end, pred, text, score}, ...]"""
    from dotenv import load_dotenv
    load_dotenv()
    from vl.voiceprint.client import VoiceprintClient, cut_audio_segment
    from difflib import SequenceMatcher

    NAME_MAP = {
        "xiyangyang": "喜羊羊", "huitailang": "灰太狼",
        "meiyangyang": "美羊羊", "lanyangyang": "懒羊羊",
        "feiyangyang": "沸羊羊", "manyangyang": "慢羊羊",
    }

    app_id = os.getenv("XFYUN_APP_ID", "")
    api_key = os.getenv("XFYUN_API_KEY", "")
    api_secret = os.getenv("XFYUN_API_SECRET", "")
    group_id = os.getenv("XFYUN_GROUP_ID", "default_group")

    client = VoiceprintClient(app_id, api_key, api_secret, group_id, verbose=False)
    tmp_dir = tempfile.mkdtemp(prefix="car_iflytek_")

    results = []
    for idx, seg in enumerate(dialogues):
        start, end = seg["start_time"], seg["end_time"]
        text = seg.get("text", "").strip()
        dur = end - start

        if dur < 0.5 or not text or text in ("(歌曲)", "(笑声)", "(音效)"):
            continue

        # 匹配 GT (仅用于后续 CAR 统计, 这里不需要)
        clip = os.path.join(tmp_dir, f"s_{idx:04d}.wav")
        try:
            cut_audio_segment(str(audio_path), start, end, clip)
            raw = client.search(clip, top_k=1)
            data = raw.get("data", {})
            sl = data.get("scoreList", [])
            if sl:
                raw_id = sl[0].get("featureId", "")
                score = sl[0].get("score", 0)
                cid = raw_id.rsplit("_", 1)[0] if "_" in raw_id else raw_id
                pred = NAME_MAP.get(cid, cid)
            else:
                pred, score = "", 0
            results.append({
                "start": start, "end": end,
                "pred": pred, "text": text, "score": round(score, 4),
            })
        except Exception:
            pass
        time.sleep(0.1)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


def main():
    print("=" * 70)
    print("Character Attribution Rate (CAR) 评估")
    print("=" * 70)

    gt = load_gt()
    print(f"GT 有效段: {len(gt)}")

    all_car = {}

    # ── Part 1: 实验8 数据 (已有讯飞结果) ──
    with open(EXP8_PATH, "r", encoding="utf-8") as f:
        exp8 = json.load(f)

    print(f"\n{'配置':<32} {'CAR':>7} {'正确':>5} {'无重叠':>5} {'集中度':>7}")
    print("=" * 70)

    for config in sorted(exp8.keys()):
        car, details = compute_car_for_config(config, gt, exp8)
        if car is None:
            continue

        correct = sum(1 for d in details if d["correct"])
        no_overlap = sum(1 for d in details if d.get("reason") == "no_overlap")
        by_spk, by_dur, avg_conc = analyze_details(details, gt)

        all_car[config] = {
            "car": round(car, 4),
            "correct": correct,
            "total": len(gt),
            "no_overlap": no_overlap,
            "concentration": round(avg_conc, 4),
            "by_speaker": {k: dict(v) for k, v in by_spk.items()},
            "by_duration": {k: dict(v) for k, v in by_dur.items()},
        }

        print(f"{config:<32} {car:>6.1%} {correct:>4}/{len(gt):<3} {no_overlap:>5} {avg_conc:>7.2%}")

    # ── Part 2: 实验9 短 chunk (需重新跑讯飞) ──
    AUDIO_PATH = PROJECT / "data" / "output" / "v3" / "052 鸟蛋之争" / "audio.wav"
    exp9_dialogues = {
        "flash_5s": EXP9_VP_DIR / "exp9_dialogues_flash_5s.json",
        "flash_10s": EXP9_VP_DIR / "exp9_dialogues_flash_10s.json",
        "flash_15s": EXP9_VP_DIR / "exp9_dialogues_flash_15s.json",
    }

    print(f"\n{'─' * 70}")
    print("实验9 短 Chunk (需重新跑讯飞)")
    print(f"{'─' * 70}")

    for label, path in sorted(exp9_dialogues.items()):
        if not path.exists():
            print(f"  {label}: 文件不存在, 跳过")
            continue

        with open(path, "r", encoding="utf-8") as f:
            dialogues = json.load(f)

        print(f"\n  [{label}] {len(dialogues)} 段, 讯飞识别中...")
        omni_segs = run_iflytek_on_omni(dialogues, gt, str(AUDIO_PATH))
        print(f"  讯飞完成: {len(omni_segs)} 段有预测")

        car, details = compute_car(gt, omni_segs)
        correct = sum(1 for d in details if d["correct"])
        no_overlap = sum(1 for d in details if d.get("reason") == "no_overlap")
        by_spk, by_dur, avg_conc = analyze_details(details, gt)

        config_name = f"flash_{label}"
        all_car[config_name] = {
            "car": round(car, 4),
            "correct": correct,
            "total": len(gt),
            "no_overlap": no_overlap,
            "concentration": round(avg_conc, 4),
            "by_speaker": {k: dict(v) for k, v in by_spk.items()},
            "by_duration": {k: dict(v) for k, v in by_dur.items()},
        }

        print(f"  CAR={car:.1%} ({correct}/{len(gt)})  无重叠={no_overlap}  集中度={avg_conc:.2%}")

    # ── 保存全部结果 ──
    out_path = PROJECT / "tests" / "experiment_results" / "exp_car.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(all_car, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_path}")

    # ── 完整对比表 ──
    print("\n" + "=" * 90)
    print("完整 CAR 对比 (Flash 系列, 按 chunk 排序)")
    print("=" * 90)
    flash_order = [
        "flash_flash_5s", "flash_flash_10s", "flash_flash_15s",
        "qwen3.5-omni-flash_30s", "qwen3.5-omni-flash_60s",
        "qwen3.5-omni-flash_90s", "qwen3.5-omni-flash_120s",
        "qwen3.5-omni-flash_150s", "qwen3.5-omni-flash_180s",
        "qwen3.5-omni-flash_300s",
    ]
    print(f"{'配置':<32} {'CAR':>7} {'正确':>6} {'无重叠':>5} {'集中度':>7}")
    print("-" * 70)
    for config in flash_order:
        if config not in all_car:
            continue
        r = all_car[config]
        print(f"{config:<32} {r['car']:>6.1%} {r['correct']:>4}/{r['total']:<3} {r['no_overlap']:>5} {r['concentration']:>7.2%}")


if __name__ == "__main__":
    main()
