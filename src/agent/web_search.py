"""Tavily 联网搜索封装 (专为 LLM 设计, 返回结构化结果).

环境变量: TAVILY_API_KEY (https://tavily.com 注册, 免费 1000 次/月)
"""

from __future__ import annotations

from src.core.config import get_config


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """搜索 query, 返回 [{title, url, content, score}, ...].

    Raises:
        RuntimeError: TAVILY_API_KEY 未配置 或 搜索失败。
    """
    api_key = get_config().tavily_api_key
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY 未配置 (https://tavily.com 注册, 免费额度)"
        )

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",  # 深度搜索, 结果更全
        include_answer=False,  # 不要 LLM 摘要, 只要原始网页结果
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            # content 截断到 400 字, 避免 prompt 过长 + reasoning 太重
            "content": (r.get("content", "") or "")[:400],
            "score": round(float(r.get("score", 0)), 3),
        }
        for r in resp.get("results", [])
    ]
