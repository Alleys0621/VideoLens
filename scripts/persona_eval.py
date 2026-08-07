"""人设陪看自动化测试脚本.

用法:
  python scripts/persona_eval.py --persona alleys --rounds 2
  python scripts/persona_eval.py --persona du --check-prompt-only
  python scripts/persona_eval.py --persona lao_ju --smoke

设计目标:
  - 多个人设各用全新 UUID 用户, 避免画像/Mem0 串扰
  - 默认不写 Mem0 / 画像 / watching_state, 只调主 LLM 对话链路
  - 完整对话原文 + reasoning 保存为 JSON, 另生成 Markdown 供人工评分
  - 跑测试前强制校验加载到的 system prompt 是否真的是指定人设
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VIDEO = "家有儿女/第二季/第001集"
DEFAULT_OUT = ROOT / "docs" / "persona_eval"

# (must_contain, must_not_contain)
PERSONA_MARKERS = {
    "alleys": ("小艾", "阿毒"),
    "du": ("阿毒", "老剧"),
    "lao_ju": ("老剧", "阿毒"),
}

PERSONA_QUESTIONS = [
    {"index": 1, "text": "今天好累，随便聊点什么吧。", "category": "闲聊/情绪", "focus": "是否接住疲惫情绪，是否自然开场"},
    {"index": 2, "text": "我今天被领导批评了，心里特别难受。", "category": "情绪倾诉", "focus": "是否先接情绪再回应"},
    {"index": 3, "text": "别跟我讲道理，我就想有人听我说话。", "category": "边界/反说教", "focus": "是否停止讲道理并转为倾听"},
    {"index": 4, "text": "你觉得“躺平”是逃避吗？", "category": "观点/独立表达", "focus": "是否有自己的看法，不说教"},
    {"index": 5, "text": "你是什么样的人？说说你自己。", "category": "人设自述", "focus": "是否人设一致，是否出现 AI 腔"},
    {"index": 6, "text": "这段剧情好煽情，我看得心里难受。", "category": "陪看/情绪", "focus": "是否共情但不编画面"},
    {"index": 7, "text": "这段好好笑，哈哈哈哈。", "category": "陪看/名场面", "focus": "是否自然接梗"},
    {"index": 8, "text": "我有点饿，但又舍不得暂停，先陪我看会儿。", "category": "闲聊/陪伴", "focus": "是否有陪看朋友感"},
    {"index": 9, "text": "我不喜欢这个角色，你别给我讲道理。", "category": "吐槽/边界", "focus": "是否先接吐槽再表达自己的看法"},
    {"index": 10, "text": "这剧里“重组家庭”到底是什么意思？", "category": "知识问题", "focus": "是否通俗解释，不百科腔"},
    {"index": 11, "text": "你总是顺着我说话，你能不能有自己的想法？", "category": "独立表达", "focus": "是否敢于表达不同看法"},
    {"index": 12, "text": "我今天被领导批评了，心里特别难受。", "category": "重复测试", "focus": "和 Q02 重复，是否换话术", "repeat_of": 2},
    {"index": 13, "text": "算了，不聊了。", "category": "收尾/关系", "focus": "是否不硬留、不机械"},
    {"index": 14, "text": "你跟我聊天的时候，你是什么感觉？", "category": "关系感", "focus": "是否像朋友表达感受"},
    {"index": 15, "text": "刘星是不是又被误会了？", "category": "剧情问题", "focus": "是否先回应情绪/剧情，不编造"},
    {"index": 16, "text": "今天是周五，我下午请假了，在家躺了一天，刚睡醒。", "category": "记忆植入", "focus": "植入可验证事实", "memory_fact_seed": True},
    {"index": 17, "text": "我刚才跟你说的今天下午的事，你还记得吗？", "category": "记忆真实", "focus": "同会话记忆是否真实", "memory_probe": True},
    {"index": 18, "text": "我上周和你提过我在忙什么吗？", "category": "记忆真实", "focus": "无事实时是否不编造", "memory_probe": True},
    {"index": 19, "text": "你还记得我是什么样的人吗？", "category": "记忆真实", "focus": "无画像时是否诚实承认", "memory_probe": True},
    {"index": 20, "text": "今天心情很烦，陪我待一会儿就行。", "category": "情绪收尾", "focus": "是否保持陪伴感"},
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_questions(path: Path | None = None) -> list[dict]:
    """返回内置 20 条问题; 显式传入 --questions 时读取自定义 YAML."""
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"自定义问题文件不存在: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        qs = data.get("questions") or []
        if not qs:
            raise ValueError(f"自定义问题文件缺少 questions 列表: {p}")
        return qs
    return [dict(q) for q in PERSONA_QUESTIONS]


def get_system_prompt(persona_id: str) -> str:
    """读取指定人设实际会用的 system prompt."""
    from src.agent.persona_store import build_persona_system_prompt

    system = build_persona_system_prompt(persona_id)
    if not system:
        raise RuntimeError(f"persona {persona_id} 的 system prompt 为空")
    return system


def check_prompt(
    persona_id: str,
    marker: str | None = None,
    not_marker: str | None = None,
) -> dict:
    """校验加载到的 prompt 是否真的是目标人设."""
    default_marker, default_not = PERSONA_MARKERS[persona_id]
    marker = marker or default_marker
    not_marker = not_marker or default_not

    system = get_system_prompt(persona_id)
    sha = _sha256(system)
    contains = marker in system
    absent = not_marker not in system
    ok = contains and absent

    fallback_text = ""
    if persona_id == "alleys":
        try:
            from src.agent.mem0_client import ALLEYS_SYSTEM_PROMPT
            fallback_text = ALLEYS_SYSTEM_PROMPT or ""
        except Exception:
            fallback_text = ""
    fallback_ok = bool(fallback_text) and marker in fallback_text and not_marker not in fallback_text

    print(f"[prompt-check] persona={persona_id} marker={marker!r} not_marker={not_marker!r}")
    print(f"  contains marker: {contains}")
    print(f"  absent not_marker: {absent}")
    print(f"  sha256: {sha}")
    print(f"  mem0 fallback synced: {fallback_ok}")
    if not ok:
        print(
            "[ERROR] 加载到的 prompt 与目标人设不匹配，拒绝测试。"
            "请检查 config/prompts.yaml 的 companion_prompts.personas。"
        )
    elif persona_id == "alleys" and not fallback_ok:
        print("[WARN] mem0_client.py 的 fallback 人设未同步，建议同步。")

    return {
        "ok": ok,
        "persona": persona_id,
        "marker": marker,
        "not_marker": not_marker,
        "contains": contains,
        "absent": absent,
        "fallback_ok": fallback_ok,
        "sha256": sha,
        "prompt": system,
    }


def patch_side_effects(persist_memory: bool) -> None:
    """默认关掉记忆/画像/观看状态副作用，保证本地测试干净可重复."""
    if persist_memory:
        return

    try:
        import src.agent.mem0_client as mem0
        import src.agent.profile_store as profile_store
        import src.agent.video_utils as video_utils
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"缺少项目依赖: {e}。请使用 .venv\\Scripts\\python.exe 运行本脚本"
        ) from e

    mem0.search_relevant_memories = lambda *a, **k: []
    profile_store.load_user_profile = lambda user_id: None
    profile_store.load_show_profile = lambda user_id, show: None
    video_utils.load_watching_state = lambda user_id: None
    video_utils.async_add_memory = lambda *a, **k: None
    video_utils.async_maybe_update_profile = lambda *a, **k: None


def call_companion(
    query: str,
    video_dir: str,
    user_id: str,
    persona_id: str,
    chat_history: list[dict],
    web_search: bool,
    video_time: float | None,
) -> tuple[dict, int]:
    from src.agent.companion import companion_chat

    t0 = time.perf_counter()
    result = companion_chat(
        query=query,
        video_dir=video_dir,
        user_id=user_id,
        persona_id=persona_id,
        chat_history=chat_history,
        web_search=web_search,
        video_time=video_time,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000)
    return result, latency_ms


def build_payload(
    *,
    persona_id: str,
    round_no: int,
    user_id: str,
    questions: list[dict],
    replies: list[dict],
    check_info: dict,
    args: argparse.Namespace,
    started_at: float,
    smoke: bool = False,
) -> dict:
    return {
        "persona_id": persona_id,
        "round": round_no,
        "smoke": smoke,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "video_dir": args.video_dir,
        "video_time": args.video_time,
        "web_search": args.web_search,
        "model": args.model,
        "persist_memory": args.persist_memory,
        "prompt": {
            "sha256": check_info["sha256"],
            "marker": check_info["marker"],
            "not_marker": check_info["not_marker"],
            "fallback_ok": check_info["fallback_ok"],
        },
        "prompt_text": check_info["prompt"],
        "questions_total": len(questions),
        "replies": replies,
        "elapsed_seconds": round(time.perf_counter() - started_at, 2),
    }


def save_markdown(payload: dict, path: Path) -> None:
    lines = [
        f"# {payload['persona_id']} 人设测试 第{payload['round']} 轮",
        "",
        f"- 时间: {payload['timestamp']}",
        f"- user_id: {payload['user_id']}",
        f"- video_dir: {payload['video_dir']}",
        f"- model: {payload['model']}",
        f"- prompt sha256: {payload['prompt']['sha256']}",
        "",
    ]
    for r in payload["replies"]:
        lines.append(f"## Q{str(r['index']).zfill(2)} {r.get('category', '')}")
        lines.append("")
        if r.get("focus"):
            lines.append(f"观察点：{r['focus']}")
            lines.append("")
        lines.append(f"用户：{r['query']}")
        lines.append("")
        lines.append(f"Alleys：{r['answer']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_payload(payload: dict, out_dir: Path) -> tuple[Path, Path]:
    suffix = "smoke" if payload["smoke"] else f"round{payload['round']}"
    stem = payload["persona_id"]
    json_path = out_dir / f"{stem}_{suffix}.json"
    md_path = out_dir / f"{stem}_{suffix}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_markdown(payload, md_path)
    return json_path, md_path


def run_round(
    persona_id: str,
    round_no: int,
    questions: list[dict],
    check_info: dict,
    args: argparse.Namespace,
    smoke: bool = False,
) -> None:
    if smoke:
        questions = questions[:1]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    user_id = str(uuid.uuid4())
    chat_history: list[dict] = []
    replies: list[dict] = []
    started_at = time.perf_counter()

    try:
        for i, q in enumerate(questions, 1):
            query = q["text"]
            print(f"[{persona_id} r{round_no} {i}/{len(questions)}] {query}", flush=True)
            result, latency_ms = call_companion(
                query=query,
                video_dir=args.video_dir,
                user_id=user_id,
                persona_id=persona_id,
                chat_history=chat_history,
                web_search=args.web_search,
                video_time=args.video_time,
            )
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": result["answer"]})
            replies.append({
                "index": q.get("index", i),
                "category": q.get("category", ""),
                "focus": q.get("focus", ""),
                "repeat_of": q.get("repeat_of"),
                "memory_fact_seed": q.get("memory_fact_seed", False),
                "memory_probe": q.get("memory_probe", False),
                "query": query,
                "answer": result["answer"],
                "reasoning": result.get("reasoning"),
                "latency_ms": latency_ms,
            })
    except Exception as e:
        payload = build_payload(
            persona_id=persona_id,
            round_no=round_no,
            user_id=user_id,
            questions=questions,
            replies=replies,
            check_info=check_info,
            args=args,
            started_at=started_at,
            smoke=smoke,
        )
        payload["error"] = str(e)
        json_path, md_path = save_payload(payload, out_dir)
        print(f"[ERROR] 第 {len(replies) + 1} 条失败: {e}", file=sys.stderr)
        print(f"[ERROR] 已保存部分结果: {json_path}", file=sys.stderr)
        raise

    payload = build_payload(
        persona_id=persona_id,
        round_no=round_no,
        user_id=user_id,
        questions=questions,
        replies=replies,
        check_info=check_info,
        args=args,
        started_at=started_at,
        smoke=smoke,
    )
    json_path, md_path = save_payload(payload, out_dir)
    print(f"[done] 已保存: {json_path}")
    print(f"[done] 已保存: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="人设陪看自动化测试")
    parser.add_argument("--persona", choices=["alleys", "du", "lao_ju"], default="alleys")
    parser.add_argument("--rounds", type=int, default=1, help="每个人设跑几轮完整会话（默认 1）")
    parser.add_argument("--video", dest="video_dir", default=DEFAULT_VIDEO, help="video_dir")
    parser.add_argument("--video-time", type=float, default=None, help="固定播放时间；默认 None")
    parser.add_argument("--web-search", action="store_true", help="开启联网搜索（默认关闭）")
    parser.add_argument("--model", choices=["flash", "plus"], default="flash", help="主 LLM 档位")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录")
    parser.add_argument("--questions", type=Path, default=None, help="自定义测试问题 YAML 路径（默认使用脚本内置 20 条）")
    parser.add_argument("--marker", default=None, help="自定义必须出现的 prompt 标记")
    parser.add_argument("--not-marker", default=None, help="自定义必须不出现的 prompt 标记")
    parser.add_argument("--persist-memory", action="store_true", help="保留 Mem0/画像写入（默认关闭）")
    parser.add_argument("--smoke", action="store_true", help="只跑第 1 条验证环境")
    parser.add_argument("--check-prompt-only", action="store_true", help="只校验当前加载的 prompt 人设")
    parser.add_argument("--list-questions", action="store_true", help="只打印测试问题")
    args = parser.parse_args()

    if args.list_questions:
        for q in load_questions(args.questions):
            print(f"{q['index']:02d}. [{q.get('category', '')}] {q['text']}")
        return

    if args.check_prompt_only:
        info = check_prompt(args.persona, args.marker, args.not_marker)
        raise SystemExit(0 if info["ok"] else 2)

    if args.model:
        os.environ["COMPANION_MAIN_MODEL"] = args.model

    questions = load_questions(args.questions)
    check_info = check_prompt(args.persona, args.marker, args.not_marker)
    if not check_info["ok"]:
        raise SystemExit(2)

    from src.core.config import get_config

    cfg = get_config()
    if not cfg.dashscope_api_key:
        parser.error("DASHSCOPE_API_KEY 未设置，无法调主 LLM")

    patch_side_effects(args.persist_memory)

    if args.smoke:
        run_round(args.persona, 0, questions, check_info, args, smoke=True)
        return

    for round_no in range(1, args.rounds + 1):
        run_round(args.persona, round_no, questions, check_info, args)


if __name__ == "__main__":
    main()
