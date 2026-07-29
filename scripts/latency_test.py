"""Latency test for VideoLens agent backend.

测试专用脚本 — 不进生产，不进启动流程。
用法: .venv/Scripts/python.exe scripts/latency_test.py
"""

import io
import json
import sys
import time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

BASE = "http://127.0.0.1:2024"


def create_thread(user_id: str = "latency-test") -> str:
    r = requests.post(
        f"{BASE}/threads",
        json={"metadata": {"user_id": user_id}},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["thread_id"]


def stream_run(thread_id: str, content: str, video_dir: str | None = None):
    body = {
        "assistant_id": "agent",
        "input": {
            "messages": [
                {"role": "user", "content": content},
            ],
        },
        "config": {
            "configurable": {
                "video_dir": video_dir or "",
                "video_time": 0,
                "user_id": "latency-test",
            },
        },
        "stream_mode": ["messages"],
    }
    t0 = time.perf_counter()
    first_byte_at = None
    first_content_at = None
    last_text = ""
    metadata_events = []
    captured_reasoning = None

    with requests.post(
        f"{BASE}/threads/{thread_id}/runs/stream",
        json=body,
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                data = line[6:]
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event == "metadata":
                    metadata_events.append(obj)
                elif event in ("messages/partial", "messages/complete"):
                    items = obj if isinstance(obj, list) else [obj]
                    for item in items:
                        if isinstance(item, dict) and item.get("type") == "ai":
                            content_val = item.get("content")
                            if content_val:
                                if first_content_at is None:
                                    first_content_at = time.perf_counter()
                                last_text = content_val
                            if captured_reasoning is None:
                                captured_reasoning = item.get("additional_kwargs", {}).get("reasoning")
            else:
                event = None

    total = time.perf_counter() - t0
    return {
        "total_seconds": total,
        "ttfb_seconds": (first_byte_at - t0) if first_byte_at else None,
        "first_token_seconds": (first_content_at - t0) if first_content_at else None,
        "text": last_text,
        "metadata": metadata_events,
        "reasoning": captured_reasoning,
    }


if __name__ == "__main__":
    thread_id = create_thread()
    print(f"thread_id: {thread_id}")

    queries = [
        ("hello", None),
        ("你好呀", "家有儿女/第一季/第01集"),
        ("这集讲什么", "家有儿女/第一季/第01集"),
        ("夏东海是谁", "家有儿女/第一季/第01集"),
    ]

    for content, video_dir in queries:
        print(f"\n=== Query: {content!r} | video_dir={video_dir} ===")
        result = stream_run(thread_id, content, video_dir)
        print(f"TTFB:        {result['ttfb_seconds']:.2f}s")
        print(f"First token: {result['first_token_seconds']:.2f}s")
        print(f"Total:       {result['total_seconds']:.2f}s")
        text = result['text']
        if text:
            print(f"Text:        {text[:100]}")
        reasoning = result.get("reasoning")
        if reasoning and isinstance(reasoning, dict):
            print(f"Intent:      {reasoning.get('intent')} (conf={reasoning.get('task_confidence')})")
            print(f"Timings:     {json.dumps(reasoning.get('timings', {}), ensure_ascii=False)}")
