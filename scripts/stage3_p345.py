"""Stage 3 P3+P4+P5 完整建库: PlotArc + Video 摘要 + Global 角色/Arc.

薄 CLI 包装, 核心逻辑在 src.pipeline.stage3.p345_kb.run_p345.

输入: data/output/{video}/stage3_dryrun.json (P1+P2 产物)
产物:
  arc_updates.json / video_summary.json / stage3_kb.json
  更新 data/output/_global/{characters,global_arcs,video_summaries}.json

用法:
  python -m scripts.stage3_p345 --video "家有儿女/第001集"
  python -m scripts.stage3_p345 --video "家有儿女/第001集" --thinking   # 开启 thinking
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.stage3 import run_p345


def main():
    parser = argparse.ArgumentParser(description="Stage 3 P3+P4+P5: PlotArc + Video 摘要 + Global")
    parser.add_argument("--video", default="家有儿女/第001集", help="video_dir")
    parser.add_argument("--thinking", action="store_true", help="开启 thinking (慢但更准)")
    args = parser.parse_args()

    run_p345(args.video, enable_thinking=args.thinking)


if __name__ == "__main__":
    main()
