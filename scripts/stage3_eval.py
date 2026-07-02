"""Stage 3 单集评估编排: dry-run (可选) → 建库评估 → 用库评估.

用法:
  python -m scripts.stage3_eval --video "家有儿女/第001集"
  python -m scripts.stage3_eval --video "家有儿女/第001集" --redo-dryrun
  python -m scripts.stage3_eval --video "家有儿女/第001集" --reuse-queries

产物 (都在 data/output/{video}/):
  stage3_dryrun.json          (建库主产物, 含 cost)
  build_quality_report.json   (建库字段完整性)
  retrieval_queries.json      (LLM 生成的 query 集)
  retrieval_eval.json         (用库 Recall@K / MRR)
  stage3_eval_summary.json    (三份报告汇总)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_config
from src.core.helpers.json_utils import load_json, save_json
from src.core.logging import get_logger
from src.eval.stage3_build_quality import evaluate as eval_build
from src.eval.stage3_retrieval import main as run_retrieval_eval

logger = get_logger()


def _print_summary_table(summary: dict) -> None:
    """在 stdout 打印关键指标表."""
    print("\n" + "=" * 70)
    print("Stage 3 单集评估汇总")
    print("=" * 70)

    print("\n[1] 规模 / 成本")
    cost = summary.get("cost", {})
    stats = summary.get("stats", {})
    print(f"  actions={stats.get('n_actions', '?')}  events={stats.get('n_events', '?')}")
    print(f"  P1 调用 {stats.get('n_p1_calls', '?')} 次  P2 调用 {stats.get('n_p2_calls', '?')} 次")
    print(f"  总成本 {cost.get('total_cost_cny', '?')} 元  "
          f"输入 {cost.get('total_input_tokens', '?')} tok  "
          f"输出 {cost.get('total_output_tokens', '?')} tok  "
          f"耗时 {cost.get('total_latency_s', '?')} s")

    print("\n[2] 建库字段完整性 (宽松)")
    bq = summary.get("build_quality", {})
    afc = bq.get("action_field_completeness", {})
    efc = bq.get("event_field_completeness", {})
    print(f"  Action 必填: " + "  ".join(f"{k}={v:.3f}" for k, v in afc.items()))
    print(f"  Event   必填: " + "  ".join(f"{k}={v:.3f}" for k, v in efc.items()))
    print(f"  Schema 违反: {bq.get('summary', {}).get('n_schema_violations', '?')}")

    print("\n[3] 建库软质量")
    soft = bq.get("quality_soft_metrics", {})
    for k, v in soft.items():
        print(f"  {k}: {v}")
    cov = bq.get("coverage", {})
    print(f"  覆盖: " + "  ".join(f"{k}={v}" for k, v in cov.items()))

    print("\n[4] 用库检索 (BM25, LLM 自生成 query)")
    rt = summary.get("retrieval", {}).get("metrics", {})
    for k, v in rt.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 单集评估编排")
    parser.add_argument("--video", default="家有儿女/第001集", help="video_dir")
    parser.add_argument("--redo-dryrun", action="store_true",
                        help="强制重跑 P1+P2 建库 (默认复用已有 stage3_dryrun.json)")
    parser.add_argument("--reuse-queries", action="store_true",
                        help="复用已有 retrieval_queries.json, 不重新调 LLM 生成")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    config = get_config()
    output_dir = os.path.join(config.output_root, args.video)
    stage3_path = os.path.join(output_dir, "stage3_dryrun.json")

    # ---------- Step 0: P1+P2 建库 ----------
    if args.redo_dryrun or not os.path.isfile(stage3_path):
        logger.info("stage3_dryrun.json 不存在或 --redo-dryrun, 先跑建库 (P1+P2)...")
        from src.pipeline.stage3 import run_p1p2
        run_p1p2(args.video, save=True)

    # ---------- Step 1: 建库评估 ----------
    logger.info("跑建库评估...")
    stage3_data = load_json(stage3_path)
    build_report = eval_build(stage3_data)
    save_json(build_report, os.path.join(output_dir, "build_quality_report.json"))

    # ---------- Step 2: 用库评估 ----------
    logger.info("跑用库评估 (BM25 + LLM query)...")
    retrieval_report = run_retrieval_eval(
        args.video, regenerate_queries=not args.reuse_queries
    )

    # ---------- Step 3: 汇总 ----------
    summary = {
        "video_id": args.video,
        "stats": stage3_data.get("stats", {}),
        "cost": stage3_data.get("cost", {}),
        "build_quality": build_report,
        "retrieval": retrieval_report,
    }
    summary_path = os.path.join(output_dir, "stage3_eval_summary.json")
    save_json(summary, summary_path)
    logger.info(f"汇总报告: {summary_path}")

    _print_summary_table(summary)


if __name__ == "__main__":
    main()
