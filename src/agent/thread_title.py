"""会话中文标题生成.

触发: 每个线程首条用户消息后, graph.companion_node 调用 maybe_set_thread_title.
存储: threads.custom_title (PG). 前端 /api/chat-threads 直接读.

设计:
  - 只在首条消息生成 (custom_title 为空时写), 后续不覆盖.
  - qwen-turbo 起短标题 (≤10 字, 无标点/书名号), 失败 fallback 截断首句.
  - 异步执行, 不阻塞回复.
"""

from __future__ import annotations

import re

from src.agent.profile_store import _conninfo, _is_uuid


def _sanitize_title(raw: str, query: str) -> str:
    """清理 LLM 输出: 去标点/引号/书名号/换行, 限长."""
    if not raw:
        return _fallback_title(query)
    t = raw.strip()
    # 去 markdown/引号/书名号/多余标点
    t = re.sub(r"[""“”‘’《》【】\[\]{}()\n\r]", "", t)
    t = re.sub(r"[，。！？!?,.;:；：·\-*]+", "", t)
    t = t.strip()
    if not t:
        return _fallback_title(query)
    # 限长 12 字
    return t[:12]


def _fallback_title(query: str) -> str:
    q = (query or "").strip().replace("\n", " ")
    if not q:
        return "新对话"
    return q[:10]


def generate_thread_title(query: str) -> str:
    """qwen-turbo 起一个中文短标题. 失败回退截断首句."""
    q = (query or "").strip()
    if not q:
        return "新对话"
    try:
        from src.core.llm.base_client import BaseLLMClient
        system = (
            "你给一段陪看对话起一个简短中文标题。要求: 4-10 字, 概括话题或情绪, "
            "像聊天记录标题; 不要书名号/引号/标点; 不要以'关于'开头; 只输出标题本身, 不要解释。"
        )
        user = f"用户这句话: {q}\n标题:"
        client = BaseLLMClient(model="qwen3-flash")
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=20,
            enable_thinking=False,
        )
        return _sanitize_title(raw, q)
    except Exception as e:
        print(f"[thread_title] 生成失败, 用 fallback: {e}", flush=True)
        return _fallback_title(q)


def set_thread_title_if_empty(thread_id: str, title: str) -> None:
    """写 threads.custom_title, 仅当当前为空. 没有行则忽略 (sync 应先建)."""
    if not _is_uuid(thread_id) or not title:
        return
    import psycopg
    try:
        with psycopg.connect(_conninfo()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE threads SET custom_title = %s
                       WHERE thread_id = %s
                         AND (custom_title IS NULL OR custom_title = '')""",
                    (title, thread_id),
                )
                conn.commit()
    except Exception as e:
        print(f"[thread_title] 写入失败: {e}", flush=True)


def maybe_set_thread_title(thread_id: str, query: str) -> None:
    """生成 + 写入 (供异步线程调用)."""
    title = generate_thread_title(query)
    set_thread_title_if_empty(thread_id, title)
    print(f"[thread_title] set {thread_id[:8]} -> {title}", flush=True)
