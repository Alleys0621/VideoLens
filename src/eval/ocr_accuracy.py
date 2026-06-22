"""
OCR 准确率评估器 — 对比 OCR 结果与 omni-plus 参考 ASR.

核心思想:
  Stage 2 OCR 输出每个 speaker anchor 时刻的字幕文字 (或 "无字幕").
  reference_asr.json 提供全集的高准确率逐句转录.
  通过 "时间重叠 + 字符相似度" 双重匹配, 评估:
    - hit_rate:     OCR 非空命中率 (排除 silence anchors)
    - recall:       参考 ASR 语句被 OCR 命中的比例 (是否漏字幕)
    - char_f1:      OCR 与参考文本的字符级 precision/recall/f1
    - miss_candidates: OCR=无字幕 但参考 ASR 该时刻有人声 (真漏)

输出: data/output/{video}/ocr_eval.json
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.logging import get_logger

logger = get_logger()

# 时间匹配窗口: OCR anchor 时间 ± MATCH_WINDOW 秒内的参考段都算候选
MATCH_WINDOW = 1.5
# 字符重叠阈值: 高于此才算 "命中参考"
MATCH_THRESHOLD = 0.5
# 判定 OCR=无字幕 是否为真漏: 窗口内参考段文本长度 >= MIN_REF_LEN
MIN_REF_LEN = 3


_PUNCT_RE = re.compile(r"[\s，。！？、,.!?;；:：\"'()（）\-—…·""''·]+")


def _normalize(text: str) -> str:
    """去掉标点空白, 小写化, 仅保留汉字/字母/数字."""
    if not text:
        return ""
    return _PUNCT_RE.sub("", text).lower()


def _char_set(text: str) -> dict[str, int]:
    """字符频次 (用于 jaccard-like 重叠)."""
    out: dict[str, int] = {}
    for ch in text:
        out[ch] = out.get(ch, 0) + 1
    return out


def _char_overlap(a: str, b: str) -> float:
    """字符级重叠系数 (|A∩B| / max(|A|,|B|)), 0-1."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    ca, cb = _char_set(na), _char_set(nb)
    inter = sum(min(ca[k], cb[k]) for k in ca if k in cb)
    return inter / max(len(na), len(nb))


def _contains_substring(needle: str, haystack: str, min_len: int = 4) -> bool:
    """较短串是否作为子串出现 (容忍 OCR 截断)."""
    na, nb = _normalize(needle), _normalize(haystack)
    if len(na) < min_len or len(nb) < min_len:
        return False
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    # 滑动 4-gram 匹配
    for i in range(len(longer) - 3):
        if longer[i:i + 4] in shorter:
            return True
    return False


@dataclass
class SceneEval:
    index: int
    anchor_type: str
    anchor_timestamp: float
    ocr_text: str
    anchor_text: Optional[str]
    ref_text: Optional[str]
    overlap: float
    is_miss_candidate: bool = False
    note: str = ""


@dataclass
class EpisodeEval:
    video: str
    n_scenes: int
    n_midpoint: int
    n_switch: int
    n_silence: int
    n_ocr_hit: int
    n_ocr_hit_nonsilence: int
    hit_rate: float
    hit_rate_nonsilence: float
    n_ref_segments: int
    n_ref_matched: int
    recall: float
    char_precision: float
    char_recall: float
    char_f1: float
    n_miss_candidates: int
    miss_rate: float
    scenes: list[SceneEval] = field(default_factory=list)


def _load_visual(visual_json: str) -> tuple[list[dict], dict[int, str]]:
    with open(visual_json, "r", encoding="utf-8") as f:
        v = json.load(f)
    scenes = v["scenes"]
    ocr_raw = v.get("ocr", {})
    ocr: dict[int, str] = {}
    for k, val in ocr_raw.items():
        try:
            ocr[int(k)] = val or ""
        except (ValueError, TypeError):
            continue
    return scenes, ocr


def _load_reference(ref_json: str) -> list[dict]:
    if not os.path.isfile(ref_json):
        return []
    with open(ref_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("segments", [])


def _find_ref_at(ref_segs: list[dict], t: float, window: float = MATCH_WINDOW):
    """返回时间窗口 [t-window, t+window] 内的参考段."""
    out = []
    for seg in ref_segs:
        s, e = seg.get("start_time", 0), seg.get("end_time", 0)
        if e < t - window or s > t + window:
            continue
        out.append(seg)
    return out


def _best_ref_match(ocr_text: str, candidates: list[dict]) -> tuple[Optional[dict], float]:
    """在候选参考段中找最佳字符重叠."""
    best_seg, best_ov = None, 0.0
    for seg in candidates:
        ov = _char_overlap(ocr_text, seg.get("text", ""))
        if ov > best_ov:
            best_ov, best_seg = ov, seg
    return best_seg, best_ov


def evaluate_episode(
    visual_json: str,
    reference_json: str,
    video: str = "",
) -> EpisodeEval:
    """对单集计算 OCR 评估指标."""
    scenes, ocr = _load_visual(visual_json)
    ref_segs = _load_reference(reference_json)

    n_mid = n_sw = n_sil = 0
    n_ocr_hit = 0
    n_ocr_hit_ns = 0  # 非静默 anchor 的 OCR 命中
    n_miss = 0

    # char-level precision/recall 累积
    tp = fp = fn = 0  # char-level (matched pairs)
    matched_ref_idx: set[int] = set()

    sce: list[SceneEval] = []

    for s in scenes:
        idx = s["index"]
        at = s.get("anchor_type", "midpoint")
        ts = s.get("anchor_timestamp", s.get("start_time", 0))
        anchor_text = s.get("anchor_text")
        text = ocr.get(idx, "")
        hit = bool(text) and text != "无字幕"

        if at == "midpoint": n_mid += 1
        elif at == "switch": n_sw += 1
        else: n_sil += 1

        if hit:
            n_ocr_hit += 1
            if at != "silence":
                n_ocr_hit_ns += 1

        # 静默 anchor: 不参与对比
        if at == "silence":
            sce.append(SceneEval(
                index=idx, anchor_type=at, anchor_timestamp=ts,
                ocr_text=text, anchor_text=anchor_text,
                ref_text=None, overlap=0.0, note="silence (skip)",
            ))
            continue

        # 找参考段
        cands = _find_ref_at(ref_segs, ts)
        ref_text = None
        overlap = 0.0
        is_miss = False

        if hit:
            best_seg, ov = _best_ref_match(text, cands)
            overlap = ov
            if best_seg is not None:
                ref_text = best_seg.get("text", "")
                # 字符级 TP/FP/FN (基于归一化频次)
                na = _normalize(text)
                nb = _normalize(ref_text)
                ca, cb = _char_set(na), _char_set(nb)
                for k in set(ca) | set(cb):
                    a, b = ca.get(k, 0), cb.get(k, 0)
                    tp += min(a, b)
                    if a > b: fp += a - b
                    if b > a: fn += b - a
                if ov >= MATCH_THRESHOLD:
                    try:
                        matched_ref_idx.add(ref_segs.index(best_seg))
                    except ValueError:
                        pass
        else:
            # OCR 漏了 — 看窗口内参考段是否有人声
            if cands:
                longest = max(cands, key=lambda x: len(_normalize(x.get("text", ""))))
                if len(_normalize(longest.get("text", ""))) >= MIN_REF_LEN:
                    is_miss = True
                    n_miss += 1
                    ref_text = longest.get("text", "")

        sce.append(SceneEval(
            index=idx, anchor_type=at, anchor_timestamp=ts,
            ocr_text=text, anchor_text=anchor_text,
            ref_text=ref_text, overlap=overlap,
            is_miss_candidate=is_miss,
        ))

    # 指标
    n_ns = n_mid + n_sw
    hit_rate = n_ocr_hit / max(n_mid + n_sw + n_sil, 1)
    hit_rate_ns = n_ocr_hit_ns / max(n_ns, 1)

    n_ref = len(ref_segs)
    n_ref_matched = len(matched_ref_idx)
    recall = n_ref_matched / n_ref if n_ref else 0.0

    char_p = tp / (tp + fp) if (tp + fp) else 0.0
    char_r = tp / (tp + fn) if (tp + fn) else 0.0
    char_f1 = 2 * char_p * char_r / (char_p + char_r) if (char_p + char_r) else 0.0

    miss_rate = n_miss / max(n_ns, 1)

    return EpisodeEval(
        video=video,
        n_scenes=len(scenes),
        n_midpoint=n_mid, n_switch=n_sw, n_silence=n_sil,
        n_ocr_hit=n_ocr_hit, n_ocr_hit_nonsilence=n_ocr_hit_ns,
        hit_rate=hit_rate, hit_rate_nonsilence=hit_rate_ns,
        n_ref_segments=n_ref, n_ref_matched=n_ref_matched,
        recall=recall,
        char_precision=char_p, char_recall=char_r, char_f1=char_f1,
        n_miss_candidates=n_miss, miss_rate=miss_rate,
        scenes=sce,
    )


def save_eval_report(ev: EpisodeEval, out_dir: str) -> str:
    """写 ocr_eval.json + 打印摘要. 返回输出路径."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ocr_eval.json")

    # 只导出关键指标 + miss 候选 (不导出所有 scene, 太大)
    miss_list = [
        {
            "index": s.index, "anchor_type": s.anchor_type,
            "timestamp": s.anchor_timestamp, "speaker_ref": None,
            "ref_text": s.ref_text, "anchor_text": s.anchor_text,
        }
        for s in ev.scenes if s.is_miss_candidate
    ]
    data = {
        "video": ev.video,
        "metrics": {
            "n_scenes": ev.n_scenes,
            "n_midpoint": ev.n_midpoint, "n_switch": ev.n_switch, "n_silence": ev.n_silence,
            "n_ocr_hit": ev.n_ocr_hit,
            "n_ocr_hit_nonsilence": ev.n_ocr_hit_nonsilence,
            "hit_rate": round(ev.hit_rate, 4),
            "hit_rate_nonsilence": round(ev.hit_rate_nonsilence, 4),
            "n_ref_segments": ev.n_ref_segments,
            "n_ref_matched": ev.n_ref_matched,
            "recall": round(ev.recall, 4),
            "char_precision": round(ev.char_precision, 4),
            "char_recall": round(ev.char_recall, 4),
            "char_f1": round(ev.char_f1, 4),
            "n_miss_candidates": ev.n_miss_candidates,
            "miss_rate": round(ev.miss_rate, 4),
        },
        "miss_candidates": miss_list,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    m = data["metrics"]
    print()
    print("=" * 64)
    print(f"[OCR Eval] {ev.video}")
    print("-" * 64)
    print(f"  scenes: {m['n_scenes']}  (mid={m['n_midpoint']} sw={m['n_switch']} sil={m['n_silence']})")
    print(f"  hit_rate (全):       {m['hit_rate']*100:.2f}%   ({m['n_ocr_hit']}/{m['n_scenes']})")
    print(f"  hit_rate (非静默):   {m['hit_rate_nonsilence']*100:.2f}%   ({m['n_ocr_hit_nonsilence']}/{m['n_midpoint']+m['n_switch']})")
    print(f"  recall (参考段覆盖): {m['recall']*100:.2f}%   ({m['n_ref_matched']}/{m['n_ref_segments']})")
    print(f"  char P/R/F1:         {m['char_precision']*100:.1f}% / {m['char_recall']*100:.1f}% / {m['char_f1']*100:.1f}%")
    print(f"  miss_candidates:     {m['n_miss_candidates']}   ({m['miss_rate']*100:.2f}%)")
    print("=" * 64)
    return out_path


def aggregate_episodes(evals: list[EpisodeEval]) -> dict:
    """跨集汇总 (mean / sum)."""
    if not evals:
        return {}
    n = len(evals)
    sums = {
        "hit_rate_nonsilence": sum(e.hit_rate_nonsilence for e in evals),
        "recall": sum(e.recall for e in evals),
        "char_f1": sum(e.char_f1 for e in evals),
        "char_precision": sum(e.char_precision for e in evals),
        "char_recall": sum(e.char_recall for e in evals),
        "miss_rate": sum(e.miss_rate for e in evals),
    }
    totals = {
        "n_scenes": sum(e.n_scenes for e in evals),
        "n_midpoint": sum(e.n_midpoint for e in evals),
        "n_ocr_hit": sum(e.n_ocr_hit for e in evals),
        "n_ocr_hit_nonsilence": sum(e.n_ocr_hit_nonsilence for e in evals),
        "n_ref_segments": sum(e.n_ref_segments for e in evals),
        "n_ref_matched": sum(e.n_ref_matched for e in evals),
        "n_miss_candidates": sum(e.n_miss_candidates for e in evals),
    }
    # 加权命中率 (按场景数加权)
    w_hit_ns = totals["n_ocr_hit_nonsilence"] / max(totals["n_midpoint"], 1)  # 近似
    w_recall = totals["n_ref_matched"] / max(totals["n_ref_segments"], 1)
    return {
        "n_episodes": n,
        "totals": totals,
        "mean": {k: round(v / n, 4) for k, v in sums.items()},
        "weighted": {
            "hit_rate_nonsilence": round(w_hit_ns, 4),
            "recall": round(w_recall, 4),
        },
        "per_episode": [
            {
                "video": e.video,
                "hit_rate_nonsilence": round(e.hit_rate_nonsilence, 4),
                "recall": round(e.recall, 4),
                "char_f1": round(e.char_f1, 4),
                "n_miss_candidates": e.n_miss_candidates,
            }
            for e in evals
        ],
    }


def print_aggregate(agg: dict) -> None:
    if not agg:
        return
    print()
    print("#" * 64)
    print(f"# Batch Aggregate  ({agg['n_episodes']} episodes)")
    print("#" * 64)
    m = agg["mean"]
    t = agg["totals"]
    w = agg["weighted"]
    print(f"  totals: {t['n_scenes']} scenes / {t['n_ref_segments']} ref segs / {t['n_ocr_hit']} hits")
    print(f"  mean hit_rate_ns:  {m['hit_rate_nonsilence']*100:.2f}%   weighted: {w['hit_rate_nonsilence']*100:.2f}%")
    print(f"  mean recall:       {m['recall']*100:.2f}%           weighted: {w['recall']*100:.2f}%")
    print(f"  mean char F1:      {m['char_f1']*100:.2f}%")
    print(f"  mean char P/R:     {m['char_precision']*100:.1f}% / {m['char_recall']*100:.1f}%")
    print(f"  mean miss_rate:    {m['miss_rate']*100:.2f}%   ({t['n_miss_candidates']} total)")
    print("-" * 64)
    print("  per-episode:")
    for e in agg["per_episode"]:
        print(f"    {e['video']:30s}  hit_ns={e['hit_rate_nonsilence']*100:5.1f}%  "
              f"recall={e['recall']*100:5.1f}%  f1={e['char_f1']*100:5.1f}%  "
              f"miss={e['n_miss_candidates']}")
    print("#" * 64)
