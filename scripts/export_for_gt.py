"""把 054 的 audio.json 导出成方便人工标 GT 的 CSV + Markdown 格式.

输出:
  data/output/054 羊毛节/gt_template.csv     - 表格软件可编辑
  data/output/054 羊毛节/gt_template.md      - markdown 表格, 直接编辑
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.helpers.json_utils import load_json


def main(video_dir: str = "054 羊毛节"):
    from src.core.config import get_config
    cfg = get_config()
    ep_dir = os.path.join(cfg.output_root, video_dir)
    audio_path = os.path.join(ep_dir, "audio.json")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    audio = load_json(audio_path)
    cr = audio.get("content_range", {}) or {}
    cs, ce = cr.get("start", 0.0), cr.get("end", 0.0)

    # 只导出 content 内、有非空 text 的段
    rows = []
    for s in audio.get("segments", []):
        b, e = s.get("begin_time", 0), s.get("end_time", 0)
        mid = (b + e) / 2
        if not (cs <= mid <= ce):
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        rows.append({
            "segment_id": s.get("segment_id", ""),
            "begin_time": round(b, 2),
            "end_time": round(e, 2),
            "duration": round(e - b, 2),
            "text": text,
            "emotion": s.get("emotion", ""),
            "speaker_pred": s.get("speaker_pred", ""),
            "vp_score": s.get("vp_score", 0),
            "speaker_gt": "",  # 留空等你标
        })

    # CSV
    csv_path = os.path.join(ep_dir, "gt_template.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path} ({len(rows)} 行)")

    # Markdown (分块, 避免一表过长)
    md_path = os.path.join(ep_dir, "gt_template.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {video_dir} - GT 标注模板\n\n")
        f.write(f"- 总段数: {len(rows)}\n")
        f.write(f"- content_range: [{cs}, {ce}]\n")
        f.write(f"- 讯飞 speaker_pred 分布:\n")
        from collections import Counter
        pred_dist = Counter(r["speaker_pred"] for r in rows)
        for k, v in pred_dist.most_common():
            f.write(f"  - {k or '(空)'}: {v}\n")
        f.write(f"\n在 speaker_gt 列填角色名即可. 候选角色: 喜羊羊 / 美羊羊 / 懒羊羊 / 沸羊羊 / 灰太狼 / 红太狼 / 慢羊羊 / 暖羊羊 / 其他.\n\n")
        f.write("| # | 时间 | text | pred | vp | speaker_gt |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            t = f"[{r['begin_time']:.1f}-{r['end_time']:.1f}]"
            text = r["text"].replace("|", "\\|").replace("\n", " ")[:60]
            f.write(f"| {i} | {t} | {text} | {r['speaker_pred']} | {r['vp_score']:.2f} |  |\n")
    print(f"MD:  {md_path}")
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="054 羊毛节")
    args = parser.parse_args()
    main(args.video)
