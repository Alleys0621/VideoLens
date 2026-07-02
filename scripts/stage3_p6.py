"""Stage 3 P6: 角色深度画像 (性格深层 + 行为模式带例证).

薄 CLI 包装, 核心逻辑在 src.pipeline.stage3.p6_profile.run_p6.

输入:
  - data/output/_global/characters.json (P5 产出)
  - 所有 data/output/*/stage3_dryrun.json (P1+P2 产出)

输出:
  - data/output/_global/character_profiles.json

用法:
  python -m scripts.stage3_p6
  python -m scripts.stage3_p6 --thinking   # 调 prompt 时用, 慢但结构化更好
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.stage3 import run_p6


def main():
    parser = argparse.ArgumentParser(description="Stage 3 P6: 角色深度画像")
    parser.add_argument("--thinking", action="store_true",
                        help="开启 thinking (调 prompt 阶段用, token 涨 3-4 倍)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    run_p6(enable_thinking=args.thinking)


if __name__ == "__main__":
    main()
