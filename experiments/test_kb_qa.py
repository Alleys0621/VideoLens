"""批量测试知识库问答

用法:
  python scripts/test_kb_qa.py --video-id "052 鸟蛋之争"
  python scripts/test_kb_qa.py --video-id "052 鸟蛋之争" --questions tests/qa/其他.json
"""

import argparse
import io
import json
import os
import sys
import time

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vl.core.config import get_config
from vl.core.helpers.json_utils import save_json
from vl.core.logging import get_logger
from vl.qa.knowledge_qa import load_knowledge_base, batch_answer, _format_knowledge_base

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="批量测试知识库问答")
    parser.add_argument("--video-id", required=True, help="视频 ID")
    parser.add_argument("--questions", help="问题 JSON 文件路径 (每行一个问题)")
    parser.add_argument("--output", default=None, help="结果输出路径 (默认: data/output/stage5_knowledge/{video_id}/qa_results.json)")
    args = parser.parse_args()

    config = get_config()
    output_dir = config.output_root
    video_id = args.video_id

    # 加载知识库
    logger.info("加载知识库: %s", video_id)
    try:
        kb = load_knowledge_base(video_id, output_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # 统计知识库
    video_title = list(kb.keys())[0] if kb else video_id
    phases = kb.get(video_title, {})
    event_count = sum(len(events) for events in phases.values())
    logger.info("知识库: %s, %d 个阶段, %d 个事件", video_title, len(phases), event_count)

    # 项目根目录
    project_root = os.path.join(os.path.dirname(__file__), "..")

    # 加载测试集: 优先 --questions 参数，其次 tests/qa/{video_id}.json
    questions_path = args.questions
    if not questions_path:
        default_path = os.path.join(project_root, "tests", "qa", f"{video_id}.json")
        if os.path.isfile(default_path):
            questions_path = default_path

    if questions_path:
        with open(questions_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        logger.info("从文件加载 %d 个问题: %s", len(qa_pairs), questions_path)
    else:
        logger.error("未找到测试集: tests/qa/%s.json，请创建或指定 --questions", video_id)
        sys.exit(1)

    # 批量回答
    logger.info("开始批量问答 (%d 个问题)...", len(qa_pairs))
    t0 = time.time()

    results = batch_answer(qa_pairs, video_id, output_dir, config)

    elapsed = time.time() - t0
    logger.info("批量问答完成: %d 个问题, 耗时 %.1f 秒", len(results), elapsed)

    # 输出结果
    print(f"\n{'=' * 70}")
    print(f"知识库问答测试结果: {video_title}")
    print(f"{'=' * 70}")

    success = 0
    for i, r in enumerate(results):
        print(f"\n[Q{i + 1}] {r['question']}")
        if r.get("reference"):
            print(f"[参考] {r['reference']}")
        if r["error"]:
            print(f"[ERROR] {r['error']}")
        else:
            print(f"[A{i + 1}] {r['answer']}")
            success += 1

    print(f"\n{'=' * 70}")
    print(f"完成: {success}/{len(results)} 个问题成功回答")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"{'=' * 70}")

    # 保存结果
    output_path = args.output or os.path.join(
        output_dir, "stage5_knowledge", video_id, "qa_results.json"
    )
    save_json(
        {
            "video_id": video_id,
            "video_title": video_title,
            "total_questions": len(results),
            "success_count": success,
            "elapsed_seconds": round(elapsed, 1),
            "results": results,
        },
        output_path,
    )
    logger.info("结果已保存: %s", output_path)


if __name__ == "__main__":
    main()
