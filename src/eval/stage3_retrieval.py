"""Stage 3 用库评估 — LLM 自生成 query + BM25 检索 + Recall@K / MRR.

流程:
  1. 对每个 event, 用 LLM 基于 summary + retrieval_text + keywords 反向生成
     1 个自然中文问句 (batch 生成, 1 次调用产出多个), 作为 GT.
  2. 用 jieba 分词 + 自实现 BM25 建 events 索引
     (检索 key = title + retrieval_text + keywords).
  3. 对每个 Q 检索 top-K, 记录原 event_id 是否命中.
  4. 算 Recall@1/3/5/10 + MRR.

输出:
  data/output/{video}/retrieval_queries.json   — query 集 + event_id
  data/output/{video}/retrieval_eval.json      — 指标 + 每条 Q 的检索结果
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from src.core.config import get_config
from src.core.helpers.json_utils import load_json, save_json
from src.core.helpers.text_utils import extract_json_obj
from src.core.llm.qwen_text import QwenTextClient
from src.core.logging import get_logger

logger = get_logger()

# BM25 超参 (Robertson-Sparck Jones 经验值)
BM25_K1 = 1.5
BM25_B = 0.75

# 单次 LLM 批量生成的 event 数
QUERY_BATCH_SIZE = 8

TOP_K_VALUES = [1, 3, 5, 10]


# ============================================================
# Step 1: LLM 反向生成 query
# ============================================================

QUERY_GEN_PROMPT = """你是中文情景剧检索测试集生成器。

任务: 对下面每个事件, 生成 1 个自然的中文问句, 该问句的答案就藏在这个事件里。
要求:
- 问句要像真实观众会问的 (如"刘梅在客厅里质问刘星什么?")
- 必须包含至少 1 个角色名 + 至少 1 个事件关键词 (地点/道具/动作)
- 不要直接复述 summary 原话, 要改写成提问形式
- 不要泄漏事件编号或时间戳
- 长度 10-30 字

# 输入事件 (JSON 数组)
{events_json}

# 输出硬约束
- 直接输出 JSON, 第一个字符是 `[`, 最后一个字符是 `]`
- 数量必须与输入事件数一致, 顺序一一对应
- 每个元素形如: {{"query": "...", "event_id": "..."}}

# 输出
"""


def _build_event_brief(e: dict) -> dict:
    """从完整 event 抽出 LLM 生成 query 所需的最小字段."""
    return {
        "event_id": e.get("event_id", ""),
        "title": e.get("title", ""),
        "summary": e.get("summary", ""),
        "retrieval_text": e.get("retrieval_text", ""),
        "keywords": e.get("keywords", []),
    }


def generate_queries(events: list[dict], client: QwenTextClient,
                     max_retries: int = 2) -> list[dict]:
    """批量调用 LLM 生成 query, 返回 [{query, event_id}, ...]."""
    results: list[dict] = []
    n_batches = math.ceil(len(events) / QUERY_BATCH_SIZE)
    for bi in range(n_batches):
        batch = events[bi * QUERY_BATCH_SIZE:(bi + 1) * QUERY_BATCH_SIZE]
        briefs = [_build_event_brief(e) for e in batch]
        events_json = json.dumps(briefs, ensure_ascii=False, indent=2)
        prompt = QUERY_GEN_PROMPT.replace("{events_json}", events_json)

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                raw = client.generate(prompt=prompt, stage="stage3_qgen",
                                      max_tokens=2000)
                if not raw:
                    raise RuntimeError("LLM 返回空")
                # 容忍 ```json``` 包裹
                parsed = extract_json_obj(raw)
                if isinstance(parsed, dict) and "queries" in parsed:
                    parsed = parsed["queries"]
                if not isinstance(parsed, list):
                    raise RuntimeError(f"返回非 JSON 数组, 前 200 字: {raw[:200]!r}")
                if len(parsed) != len(batch):
                    raise RuntimeError(
                        f"query 数量不一致: 期望 {len(batch)}, 实际 {len(parsed)}"
                    )
                for q in parsed:
                    if isinstance(q, dict) and q.get("query"):
                        results.append({
                            "query": q["query"],
                            "event_id": q.get("event_id") or "",
                        })
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    logger.warning(
                        f"[QGEN batch {bi+1}/{n_batches}] RETRY {attempt+1}/{max_retries}: {e}"
                    )
                    continue
        if last_err:
            logger.error(f"[QGEN batch {bi+1}/{n_batches}] 最终失败: {last_err}")
            # 失败的 batch 用 title 兜底当 query, 不让整轮评估挂掉
            for e in batch:
                title = e.get("title", "") or e.get("event_id", "")
                results.append({"query": title, "event_id": e.get("event_id", "")})

        logger.info(f"[QGEN] batch {bi+1}/{n_batches} 完成, 累计 {len(results)} queries")

    # 强制把 event_id 对齐回真实 batch 顺序 (LLM 偶尔会乱填)
    for i, e in enumerate(events[:len(results)]):
        results[i]["event_id"] = e.get("event_id", results[i].get("event_id", ""))

    return results


# ============================================================
# Step 2: BM25 索引 (jieba 分词, 自实现)
# ============================================================

def _tokenize(text: str) -> list[str]:
    """jieba 切词 + 过滤空白/单字噪声."""
    import jieba
    tokens = jieba.lcut(text or "")
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


class BM25Index:
    """简单 BM25 实现, 支持 Chinese via jieba."""

    def __init__(self, docs: list[str]):
        self.docs_tokens = [_tokenize(d) for d in docs]
        self.n_docs = len(self.docs_tokens)
        self.avg_dl = (
            sum(len(t) for t in self.docs_tokens) / self.n_docs
            if self.n_docs else 0.0
        )
        # 文档频率
        self.df: dict[str, int] = {}
        for tokens in self.docs_tokens:
            for tok in set(tokens):
                self.df[tok] = self.df.get(tok, 0) + 1
        # IDF (Robertson-Sparck Jones + 1 平滑)
        self.idf: dict[str, float] = {
            tok: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for tok, df in self.df.items()
        }
        # 各文档词频
        self.tf: list[dict[str, int]] = []
        for tokens in self.docs_tokens:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        tf_map = self.tf[doc_idx]
        dl = sum(tf_map.values())
        if dl == 0:
            return 0.0
        score = 0.0
        denom_norm = BM25_K1 * (1 - BM25_B + BM25_B * dl / (self.avg_dl or 1.0))
        for tok in query_tokens:
            if tok not in tf_map:
                continue
            tf = tf_map[tok]
            idf = self.idf.get(tok, 0.0)
            score += idf * (tf * (BM25_K1 + 1)) / (tf + denom_norm)
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_tokens = _tokenize(query)
        scored = [(i, self.score(q_tokens, i)) for i in range(self.n_docs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def build_searchable_text(e: dict) -> str:
    """拼接 event 的可检索文本."""
    parts = [
        e.get("title", ""),
        e.get("retrieval_text", ""),
        " ".join(e.get("keywords") or []),
    ]
    return " ".join(p for p in parts if p)


# ============================================================
# Step 3: 评估指标
# ============================================================

def evaluate(queries: list[dict], events: list[dict]) -> dict:
    """对所有 query 做 BM25 检索, 算 Recall@K / MRR."""
    docs = [build_searchable_text(e) for e in events]
    index = BM25Index(docs)

    eid_to_idx = {e.get("event_id", ""): i for i, e in enumerate(events)}

    per_query = []
    hit_counts = {k: 0 for k in TOP_K_VALUES}
    mrr_sum = 0.0

    max_k = max(TOP_K_VALUES)
    for q in queries:
        gt_eid = q.get("event_id", "")
        gt_idx = eid_to_idx.get(gt_eid, -1)
        if gt_idx < 0:
            logger.warning(f"query event_id={gt_eid!r} 在 events 中找不到, 跳过")
            continue

        top = index.search(q.get("query", ""), top_k=max_k)
        top_ids = [i for i, _ in top]

        ranks = top_ids.index(gt_idx) + 1 if gt_idx in top_ids else 0
        for k in TOP_K_VALUES:
            if ranks and ranks <= k:
                hit_counts[k] += 1
        if ranks:
            mrr_sum += 1.0 / ranks

        per_query.append({
            "query": q.get("query", ""),
            "gt_event_id": gt_eid,
            "rank": ranks,
            "top_5_event_ids": [events[i].get("event_id", "") for i in top_ids[:5]],
            "top_5_scores": [round(s, 4) for _, s in top[:5]],
        })

    n_valid = len(per_query)
    metrics = {
        "n_queries": n_valid,
        **{f"recall@{k}": round(hit_counts[k] / n_valid, 4) if n_valid else 0.0
           for k in TOP_K_VALUES},
        "mrr": round(mrr_sum / n_valid, 4) if n_valid else 0.0,
    }

    return {
        "metrics": metrics,
        "per_query": per_query,
    }


# ============================================================
# 入口
# ============================================================

def main(video_dir: str, regenerate_queries: bool = True) -> dict:
    """跑完整用库评估. 返回 retrieval_eval 报告 dict."""
    config = get_config()
    output_dir = os.path.join(config.output_root, video_dir)
    stage3_path = os.path.join(output_dir, "stage3_dryrun.json")
    if not os.path.isfile(stage3_path):
        raise FileNotFoundError(f"未找到 {stage3_path}; 先跑 scripts.stage3_p1p2 --save")

    stage3_data = load_json(stage3_path)
    events = stage3_data.get("events", []) or []
    if not events:
        raise ValueError("stage3_dryrun.json 中没有 events")

    # Step 1: 生成 query
    queries_path = os.path.join(output_dir, "retrieval_queries.json")
    if regenerate_queries or not os.path.isfile(queries_path):
        client = QwenTextClient()
        queries = generate_queries(events, client)
        save_json(queries, queries_path)
        logger.info(f"query 集已保存: {queries_path}")
    else:
        queries = load_json(queries_path)
        logger.info(f"复用已有 query 集: {queries_path} ({len(queries)} 条)")

    # Step 2 + 3: 建索引 + 评估
    report = evaluate(queries, events)

    out_path = os.path.join(output_dir, "retrieval_eval.json")
    save_json(report, out_path)
    logger.info(f"用库评估报告已保存: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3 用库评估 (BM25)")
    parser.add_argument("--video", default="家有儿女/第001集", help="video_dir")
    parser.add_argument("--reuse-queries", action="store_true",
                        help="复用已有 retrieval_queries.json, 不重新生成")
    args = parser.parse_args()
    main(args.video, regenerate_queries=not args.reuse_queries)
