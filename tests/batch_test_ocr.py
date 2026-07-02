"""
批量 OCR 测试 — 7 集 家有儿女 (S1全3集 + S2E2-5).

每集流程:
  1. Stage 1: 音频 + ASR + 声纹 (生成 speaker segments 作为 anchors 输入)
  2. Stage 2: 视觉 (基于 anchors 抽帧 + qwen3-vl-plus OCR)
  3. omni-plus 参考 ASR (eval 专用, 全集高准确率转录)
  4. ocr_accuracy.evaluate_episode → ocr_eval.json

并行策略: 集间并行 (max_workers=3), 集内串行.
预估总耗时: ~1.5h, 总成本: ~60-70 CNY.

用法:
  python -m tests.batch_test_ocr                      # 跑全集 7 集
  python -m tests.batch_test_ocr --episodes "S1E1,S1E2"
  python -m tests.batch_test_ocr --skip-stage1         # 跳过 Stage 1 (已有 audio.json)
  python -m tests.batch_test_ocr --skip-reference      # 跳过 omni-plus 参考 ASR
  python -m tests.batch_test_ocr --only-eval           # 仅重算评估 (假定前面都已跑)
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

# ══════════════════════════════════════════════════════════════
# 剧集清单
# ══════════════════════════════════════════════════════════════
EPISODES = [
    # (id, video_dir, 说明)
    ("S1E1", "家有儿女/第一季/第01集", "第一季 第1集 (.mkv)"),
    ("S1E2", "家有儿女/第一季/第02集", "第一季 第2集 (.mkv)"),
    ("S1E3", "家有儿女/第一季/第03集", "第一季 第3集 (.mkv)"),
    ("S2E2", "家有儿女/第二季/第002集", "第二季 第2集 (.mp4)"),
    ("S2E3", "家有儿女/第二季/第003集", "第二季 第3集 (.mp4)"),
    ("S2E4", "家有儿女/第二季/第004集", "第二季 第4集 (.mp4)"),
    ("S2E5", "家有儿女/第二季/第005集", "第二季 第5集 (.mp4)"),
]


def process_episode(
    ep_id: str,
    video_dir: str,
    skip_stage1: bool,
    skip_reference: bool,
    only_eval: bool,
) -> dict:
    """处理单集: Stage1 → Stage2 → ref ASR → eval. 返回结果 dict."""
    from src.core.path_utils import resolve_video_path, get_show_name, load_voiceprint_config

    output_dir = os.path.join("data", "output", video_dir)
    os.makedirs(output_dir, exist_ok=True)
    visual_json = os.path.join(output_dir, "visual.json")
    audio_json = os.path.join(output_dir, "audio.json")
    ref_json = os.path.join(output_dir, "reference_asr.json")
    eval_json = os.path.join(output_dir, "ocr_eval.json")

    t0 = time.time()
    result = {
        "ep_id": ep_id, "video_dir": video_dir,
        "status": "running", "error": None,
        "stages": {},
    }

    try:
        video_path = resolve_video_path(video_dir)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")
        print(f"\n[{ep_id}] 开始  video={video_path}")

        # ── Stage 1 ─────────────────────────────────────
        audio_result = None
        if not only_eval:
            if skip_stage1 and os.path.isfile(audio_json):
                print(f"[{ep_id}] Stage 1: 跳过 (audio.json 已存在)")
                with open(audio_json, "r", encoding="utf-8") as f:
                    audio_result = json.load(f)
            else:
                from src.pipeline.stage1_audio import run_stage1
                show = get_show_name(video_dir)
                gid, nm = load_voiceprint_config(show)
                t1 = time.time()
                audio_result = run_stage1(
                    video_dir, output_dir,
                    skip_theme=False, chunk_dur=60,
                    vp_threshold=0.0, group_id=gid, name_map=nm or {},
                )
                result["stages"]["stage1_sec"] = round(time.time() - t1, 1)
                print(f"[{ep_id}] Stage 1 完成 ({result['stages']['stage1_sec']}s)")

        # ── Stage 2 ─────────────────────────────────────
        if not only_eval:
            from src.pipeline.stage2_visual import run_stage2
            t2 = time.time()
            run_stage2(video_dir, output_dir, audio_result=audio_result, skip_captions=True)
            result["stages"]["stage2_sec"] = round(time.time() - t2, 1)
            print(f"[{ep_id}] Stage 2 完成 ({result['stages']['stage2_sec']}s)")

        # ── 参考 ASR ────────────────────────────────────
        if not only_eval and not skip_reference:
            from src.eval.reference_asr import run_reference_asr
            t3 = time.time()
            run_reference_asr(
                video_dir=video_dir,
                output_dir=output_dir,
                video_path=video_path,
                chunk_duration=60.0,
                max_workers=4,
            )
            result["stages"]["ref_asr_sec"] = round(time.time() - t3, 1)
            print(f"[{ep_id}] 参考 ASR 完成 ({result['stages']['ref_asr_sec']}s)")
        elif only_eval or skip_reference:
            print(f"[{ep_id}] 跳过参考 ASR (使用现有 reference_asr.json 或留空)")

        # ── Eval ────────────────────────────────────────
        from src.eval.ocr_accuracy import evaluate_episode, save_eval_report
        ev = evaluate_episode(visual_json, ref_json, video=ep_id)
        save_eval_report(ev, output_dir)
        result["status"] = "ok"
        result["metrics"] = {
            "hit_rate_nonsilence": round(ev.hit_rate_nonsilence, 4),
            "recall": round(ev.recall, 4),
            "char_f1": round(ev.char_f1, 4),
            "char_precision": round(ev.char_precision, 4),
            "char_recall": round(ev.char_recall, 4),
            "n_miss_candidates": ev.n_miss_candidates,
            "n_midpoint": ev.n_midpoint,
            "n_switch": ev.n_switch,
            "n_silence": ev.n_silence,
            "n_ref_segments": ev.n_ref_segments,
            "n_ref_matched": ev.n_ref_matched,
        }

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"[{ep_id}] 错误: {result['error']}")
        traceback.print_exc()

    result["total_sec"] = round(time.time() - t0, 1)
    print(f"[{ep_id}] 结束  status={result['status']}  total={result['total_sec']}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="",
                    help="逗号分隔的 ep_id (如 'S1E1,S2E3'), 留空=全7集")
    ap.add_argument("--workers", type=int, default=3,
                    help="集间并行数")
    ap.add_argument("--skip-stage1", action="store_true",
                    help="跳过 Stage 1 (使用现有 audio.json)")
    ap.add_argument("--skip-reference", action="store_true",
                    help="跳过 omni-plus 参考 ASR")
    ap.add_argument("--only-eval", action="store_true",
                    help="仅重算评估")
    args = ap.parse_args()

    eps = EPISODES
    if args.episodes:
        wanted = {e.strip() for e in args.episodes.split(",")}
        eps = [e for e in EPISODES if e[0] in wanted]
        if not eps:
            print(f"未匹配到任何 episode: {args.episodes}")
            sys.exit(1)

    print(f"# 批量 OCR 测试: {len(eps)} 集, workers={args.workers}")
    print(f"# skip_stage1={args.skip_stage1}  skip_reference={args.skip_reference}  only_eval={args.only_eval}")
    print(f"# episodes: {[e[0] for e in eps]}")
    print()

    t0 = time.time()
    results: list[dict] = []

    # 集间并行
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                process_episode, ep_id, vd,
                args.skip_stage1, args.skip_reference, args.only_eval,
            ): ep_id
            for ep_id, vd, _ in eps
        }
        for fut in as_completed(futs):
            ep_id = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"ep_id": ep_id, "status": "error",
                     "error": f"{type(e).__name__}: {e}"}
                print(f"[{ep_id}] fatal: {e}")
            results.append(r)

    # 按原顺序排序
    order = {e[0]: i for i, e in enumerate(EPISODES)}
    results.sort(key=lambda r: order.get(r["ep_id"], 999))

    # 写 batch_report.json
    batch_dir = os.path.join("data", "output", "_batch_reports")
    os.makedirs(batch_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    batch_path = os.path.join(batch_dir, f"batch_ocr_{ts}.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump({
            "episodes_run": [e[0] for e in eps],
            "options": {
                "skip_stage1": args.skip_stage1,
                "skip_reference": args.skip_reference,
                "only_eval": args.only_eval,
                "workers": args.workers,
            },
            "results": results,
            "total_sec": round(time.time() - t0, 1),
        }, f, ensure_ascii=False, indent=2)

    # ── 汇总 ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"# 批量测试完成  总耗时 {time.time()-t0:.0f}s  结果 → {batch_path}")
    print("=" * 70)
    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") != "ok"]
    print(f"# 成功 {len(ok)}/{len(results)}  失败 {len(err)}")
    if err:
        for r in err:
            print(f"  FAIL {r['ep_id']}: {r.get('error','?')}")

    if ok:
        # 用 evaluate_episode 重新构造 EpisodeEval 来用 aggregate
        from src.eval.ocr_accuracy import (
            EpisodeEval, aggregate_episodes, print_aggregate,
        )
        evs = []
        for r in ok:
            m = r["metrics"]
            evs.append(EpisodeEval(
                video=r["ep_id"],
                n_scenes=m["n_midpoint"] + m["n_switch"] + m["n_silence"],
                n_midpoint=m["n_midpoint"], n_switch=m["n_switch"],
                n_silence=m["n_silence"],
                n_ocr_hit=0, n_ocr_hit_nonsilence=0,
                hit_rate=0.0,
                hit_rate_nonsilence=m["hit_rate_nonsilence"],
                n_ref_segments=m["n_ref_segments"],
                n_ref_matched=m["n_ref_matched"],
                recall=m["recall"],
                char_precision=m["char_precision"],
                char_recall=m["char_recall"],
                char_f1=m["char_f1"],
                n_miss_candidates=m["n_miss_candidates"],
                miss_rate=m["n_miss_candidates"] / max(m["n_midpoint"] + m["n_switch"], 1),
            ))
        agg = aggregate_episodes(evs)
        print_aggregate(agg)

        # 保存 aggregate
        agg_path = os.path.join(batch_dir, f"batch_ocr_{ts}_aggregate.json")
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print(f"# 跨集汇总 → {agg_path}")

    # Cost
    try:
        from src.core.cost import get_cost_tracker
        print("\n" + get_cost_tracker().report())
    except Exception as e:
        print(f"(cost report 失败: {e})")


if __name__ == "__main__":
    main()
