"""Stage 3 P1+P2 单集建库: Action 抽取 + Event 聚合.

薄 CLI 包装, 核心逻辑在 src.pipeline.stage3.p1_p2_actions.run_p1p2.

产物 (data/output/{video}/):
  stage3_dryrun.json — P1+P2 结果 (actions + events + stats + cost)

用法:
  python -m scripts.stage3_p1p2 --video "家有儿女/第001集"
  python -m scripts.stage3_p1p2 --video "家有儿女/第001集" --limit-p1 30   # 快速验证
  python -m scripts.stage3_p1p2 --video "家有儿女/第001集" --save          # 保存结果
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.stage3 import run_p1p2


def main():
    parser = argparse.ArgumentParser(description="Stage 3 P1+P2: Action 抽取 + Event 聚合")
    parser.add_argument("--video", default="家有儿女/第001集", help="video_dir (相对 output_root)")
    parser.add_argument("--p1-batch", type=int, default=5, help="P1 batch 大小 (候选数/批)")
    parser.add_argument("--p2-batch", type=int, default=8, help="P2 batch 大小 (action 数/批)")
    parser.add_argument("--limit-p1", type=int, default=0,
                        help="只跑前 N 个候选 (0=全部), 用于快速验证")
    parser.add_argument("--save", action="store_true",
                        help="保存结果到 data/output/{video}/stage3_dryrun.json")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    run_p1p2(
        args.video,
        p1_batch=args.p1_batch,
        p2_batch=args.p2_batch,
        limit_p1=args.limit_p1,
        save=args.save,
    )


if __name__ == "__main__":
    main()
