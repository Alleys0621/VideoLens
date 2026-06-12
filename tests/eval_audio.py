"""
Stage 1 音频处理评估脚本

评估指标 (仅统计主角团):
  - CAR (Character Attribution Rate): 角色归属正确率
      CAR = 主角团正确识别的 GT 段数 / 主角团 GT 段数
  - 情感准确率: 情感标签与 GT 的一致率

用法:
  # 评估指定视频的 pipeline 输出
  python -m tests.eval_audio --video "052 鸟蛋之争"

  # 指定声纹置信度阈值
  python -m tests.eval_audio --video "052 鸟蛋之争" --vp-threshold 0.4

  # 输出详细报告
  python -m tests.eval_audio --video "052 鸟蛋之争" --detail

  # 保存报告到 tests/results/
  python -m tests.eval_audio --video "052 鸟蛋之争" --save

数据依赖:
  - data/output/{video}/audio.json  (Pipeline 输出)
  - data/gt/喜羊羊与灰太狼/xiyangyang*_gt.json  (统一 GT)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

# 主角团角色名 — 从 pipeline.yaml voiceprint_groups 动态加载
def _load_main_characters():
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "pipeline.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        chars = set()
        for show_cfg in cfg.get("voiceprint_groups", {}).values():
            chars.update(show_cfg.get("name_mapping", {}).values())
        return chars
    return set()

MAIN_CHARACTERS = _load_main_characters()

# 视频目录名 → GT 文件路径 映射
GT_FILE_MAP = {
    "052 鸟蛋之争": "data/gt/喜羊羊与灰太狼/xiyangyang052_gt.json",
    "053 懒羊羊的歌声": "data/gt/喜羊羊与灰太狼/xiyangyang053_gt.json",
}


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def find_gt_path(video: str) -> str | None:
    """根据视频目录名查找 GT 文件路径"""
    if video in GT_FILE_MAP:
        return GT_FILE_MAP[video]

    gt_root = os.path.join("data", "gt")

    # 子目录格式: "家有儿女/第001集" → data/gt/家有儿女/001_gt.json
    if "/" in video or "\\" in video:
        parts = video.replace("\\", "/").split("/")
        show_dir = parts[0]
        ep_name = parts[-1]
        show_gt_dir = os.path.join(gt_root, show_dir)
        if os.path.isdir(show_gt_dir):
            for f in os.listdir(show_gt_dir):
                if f.endswith("_gt.json"):
                    return os.path.join(show_gt_dir, f)

    # 平铺格式: 用集号搜索
    ep_num = video.split()[0] if video else ""
    for root, dirs, files in os.walk(gt_root):
        for f in files:
            if f.endswith("_gt.json") and ep_num in f:
                return os.path.join(root, f)
    return None


def load_gt(video: str) -> list[dict]:
    """加载统一 GT (speaker_gt + emotion_gt)"""
    gt_path = find_gt_path(video)
    if not gt_path or not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"未找到 GT 文件，请确认 data/gt/ 下存在 {video} 对应的 GT 文件"
        )
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容两种格式: 直接数组 或 {"segments": [...]}
    if isinstance(data, list):
        return data
    return data.get("segments", data)


def load_pipeline_output(output_dir: str) -> dict:
    """加载 Pipeline 输出的 audio.json"""
    audio_path = os.path.join(output_dir, "audio.json")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Pipeline 输出不存在: {audio_path}\n"
            f"请先运行: python -m src.app.main run \"{os.path.basename(output_dir)}\" --stage 1"
        )
    with open(audio_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# 时间重叠
# ══════════════════════════════════════════════════════════════

def overlap_duration(s1, e1, s2, e2):
    return max(0.0, min(e1, e2) - max(s1, s2))


# ══════════════════════════════════════════════════════════════
# CAR: Character Attribution Rate
# ══════════════════════════════════════════════════════════════

def compute_car(gt_main, pred_segments):
    """CAR = 正确识别段数 / 有预测重叠的 GT 段数 (漏识别不纳入分母)"""
    details = []

    for gi, g in enumerate(gt_main):
        gs, ge = g["begin_time"], g["end_time"]
        gt_spk = g["speaker_gt"]

        contributions = defaultdict(float)
        for seg in pred_segments:
            pred = seg.get("speaker_pred", "")
            if not pred:
                continue
            ov = overlap_duration(gs, ge, seg["begin_time"], seg["end_time"])
            if ov <= 0:
                continue
            seg_dur = seg["end_time"] - seg["begin_time"]
            if seg_dur <= 0:
                continue
            contributions[pred] += (ov / seg_dur) * len(seg.get("text", ""))

        if not contributions:
            # 漏识别: 不纳入统计
            continue

        predicted = max(contributions, key=contributions.get)
        details.append({
            "gt_idx": gi, "gt_speaker": gt_spk, "gt_text": g["text"][:40],
            "gt_duration": round(ge - gs, 1),
            "predicted": predicted, "correct": predicted == gt_spk,
        })

    correct = sum(1 for d in details if d["correct"])
    car = correct / len(details) if details else 0.0
    return car, details


# ══════════════════════════════════════════════════════════════
# 情感准确率
# ══════════════════════════════════════════════════════════════

def compute_emotion_accuracy(gt_segments, pred_segments):
    """情感识别准确率 (全部 GT 段)"""
    details = []
    for gi, g in enumerate(gt_segments):
        gt_emotion = g.get("emotion_gt", "")
        if not gt_emotion:
            continue
        gs, ge = g["begin_time"], g["end_time"]

        best_ov, best_emotion = 0.0, ""
        for seg in pred_segments:
            ov = overlap_duration(gs, ge, seg["begin_time"], seg["end_time"])
            if ov > best_ov:
                best_ov = ov
                best_emotion = seg.get("emotion", "")

        if best_ov <= 0:
            continue
        details.append({
            "gt_idx": gi, "gt_speaker": g.get("speaker_gt", ""),
            "gt_emotion": gt_emotion, "predicted": best_emotion,
            "correct": best_emotion == gt_emotion,
        })

    if not details:
        return 0.0, []
    correct = sum(1 for d in details if d["correct"])
    return correct / len(details), details


# ══════════════════════════════════════════════════════════════
# 细粒度分析
# ══════════════════════════════════════════════════════════════

def analyze_by_speaker(details):
    by_spk = defaultdict(lambda: {"total": 0, "correct": 0})
    for d in details:
        spk = d["gt_speaker"]
        by_spk[spk]["total"] += 1
        if d["correct"]:
            by_spk[spk]["correct"] += 1
    return {k: dict(v) for k, v in sorted(by_spk.items())}


def analyze_by_duration(details, gt_segments):
    by_dur = defaultdict(lambda: {"total": 0, "correct": 0})
    for d in details:
        gi = d["gt_idx"]
        dur = gt_segments[gi]["end_time"] - gt_segments[gi]["begin_time"]
        label = "<2s" if dur < 2 else "2-5s" if dur < 5 else "5-10s" if dur < 10 else ">10s"
        by_dur[label]["total"] += 1
        if d["correct"]:
            by_dur[label]["correct"] += 1
    return {k: dict(v) for k, v in sorted(by_dur.items())}


# ══════════════════════════════════════════════════════════════
# 报告输出
# ══════════════════════════════════════════════════════════════

def build_report(video, car, car_details, emotion_acc, emotion_details,
                 pred_segments, gt_all, gt_main, vp_threshold, show_detail):
    """构建报告文本行列表"""
    gt_extra = [g for g in gt_all if g["speaker_gt"] not in MAIN_CHARACTERS]
    with_pred = sum(1 for s in pred_segments if s.get("speaker_pred"))
    coverage = with_pred / len(pred_segments) if pred_segments else 0

    car_correct = sum(1 for d in car_details if d["correct"])
    matched = len(car_details)
    no_match = len(gt_main) - matched

    lines = []
    lines.append("=" * 60)
    lines.append(f"Stage 1 音频评估: {video}")
    if vp_threshold > 0:
        lines.append(f"声纹阈值: {vp_threshold}")
    lines.append("=" * 60)
    lines.append(f"Pipeline: {len(pred_segments)} 段, 声纹预测: {with_pred} ({coverage:.1%})")
    lines.append(f"GT: {len(gt_all)} 段 (主角团 {len(gt_main)}, 其他 {len(gt_extra)})")
    lines.append(f"")
    lines.append(f"CAR: {car:.1%} ({car_correct}/{matched})")
    if no_match > 0:
        lines.append(f"漏识别: {no_match} 段 (不纳入分母)")
    if emotion_details:
        emo_correct = sum(1 for d in emotion_details if d["correct"])
        lines.append(f"情感: {emotion_acc:.1%} ({emo_correct}/{len(emotion_details)})")

    by_spk = analyze_by_speaker(car_details)
    if by_spk:
        lines.append(f"\n按角色 (CAR):")
        for spk, v in sorted(by_spk.items(), key=lambda x: -x[1]["total"]):
            rate = v["correct"] / v["total"] if v["total"] > 0 else 0
            lines.append(f"  {spk}: {v['correct']}/{v['total']} ({rate:.1%})")

    by_dur = analyze_by_duration(car_details, gt_main)
    if by_dur:
        lines.append(f"\n按时长 (CAR):")
        for label in ["<2s", "2-5s", "5-10s", ">10s"]:
            if label in by_dur:
                v = by_dur[label]
                rate = v["correct"] / v["total"] if v["total"] > 0 else 0
                lines.append(f"  {label}: {v['correct']}/{v['total']} ({rate:.1%})")

    # 错误详情
    errors = [d for d in car_details if not d["correct"]]
    if errors:
        lines.append(f"\n错误详情 ({len(errors)} 段):")
        for d in errors:
            lines.append(f"  [{d['gt_idx']}] GT={d['gt_speaker']} Pred={d['predicted']} | {d['gt_text']}")

    # 详细报告
    if show_detail:
        lines.append(f"\n逐段匹配:")
        for d in car_details:
            mark = "Y" if d["correct"] else "X"
            lines.append(f"  {d['gt_idx']:>3} {d['gt_speaker']:<8} {d['predicted']:<8} {mark} {d['gt_text']}")

    lines.append(f"\n{'=' * 60}")
    return lines


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 音频处理评估 (声纹 + 情感)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m tests.eval_audio --video "052 鸟蛋之争"
  python -m tests.eval_audio --video "052 鸟蛋之争" --vp-threshold 0.4
  python -m tests.eval_audio --video "052 鸟蛋之争" --detail --save
        """,
    )
    parser.add_argument("--video", required=True, help="视频目录名")
    parser.add_argument("--vp-threshold", type=float, default=0.0,
                        help="声纹置信度阈值, 低于此值标为 '路人' (默认: 0)")
    parser.add_argument("--detail", action="store_true", help="输出逐段匹配详情")
    parser.add_argument("--save", action="store_true", help="保存报告到 tests/results/")
    args = parser.parse_args()

    video_dir = args.video
    output_dir = os.path.join("data", "output", video_dir)

    # 加载 Pipeline 输出
    pipeline_data = load_pipeline_output(output_dir)
    pred_segments = pipeline_data.get("segments", [])

    # 应用声纹阈值
    if args.vp_threshold > 0:
        for seg in pred_segments:
            if seg.get("speaker_pred") and seg.get("vp_score", 0) < args.vp_threshold:
                seg["speaker_pred"] = "路人"

    # 加载 GT
    gt_all = load_gt(video_dir)
    gt_main = [g for g in gt_all if g["speaker_gt"] in MAIN_CHARACTERS]

    if not gt_main:
        print(f"错误: GT 中无主角团段。角色: {set(g['speaker_gt'] for g in gt_all)}")
        sys.exit(1)

    # 计算 CAR
    car, car_details = compute_car(gt_main, pred_segments)

    # 计算情感准确率
    emotion_acc, emotion_details = compute_emotion_accuracy(gt_all, pred_segments)

    # 构建报告
    lines = build_report(video_dir, car, car_details, emotion_acc, emotion_details,
                         pred_segments, gt_all, gt_main, args.vp_threshold, args.detail)

    # 打印
    for line in lines:
        print(line)

    # 保存
    if args.save:
        results_dir = os.path.join("tests", "results")
        save_path = os.path.join(results_dir, f"eval_{video_dir.replace(' ', '_').replace('/', '_').replace(os.sep, '_')}.txt")
        os.makedirs(results_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n报告已保存: {save_path}")


if __name__ == "__main__":
    main()
