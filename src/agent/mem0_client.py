"""陪看智能体「Alleys」: 基于 Mem0 的对话记忆 + 陪看人设

设计:
  - 短期记忆: Streamlit session_state (本会话内)
  - 长期记忆: Mem0 (跨会话持久化, 存到 data/_mem0_qdrant/)
  - 陪看人设: 「Alleys」陪看搭子, 25 岁女生, 喜欢看剧
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# 让 Mem0 内部的 OpenAI client 走 DashScope
if os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ============================================================
# Mem0 单例 (懒加载)
# ============================================================

_mem0_instance = None


def get_memory():
    """获取 Mem0 实例 (单例, 避免重复加载 BGE 模型)."""
    global _mem0_instance
    if _mem0_instance is not None:
        return _mem0_instance
    from mem0 import Memory
    config = {
        "llm": {
            "provider": "openai",
            "config": {"model": "qwen3.7-plus", "temperature": 0.1},
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-v3",
                "api_key": os.getenv("DASHSCOPE_API_KEY"),
                "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(PROJECT_ROOT / "data" / "_mem0_qdrant"),
                "embedding_model_dims": 1024,
            },
        },
        "custom_instructions": (
            "所有记忆必须用中文存储, 不要翻译成英文. "
            "记录用户的观影偏好、对角色的看法、聊过的剧情点. "
            "每条记忆简洁, 不超过 30 字."
        ),
        "version": "v1.1",
    }
    _mem0_instance = Memory.from_config(config)
    return _mem0_instance


# ============================================================
# 记忆 API: add / search
# ============================================================

def add_conversation_memory(
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    video_id: str | None = None,
) -> dict | None:
    """把一轮对话存入 Mem0 (LLM 自动提炼要点).

    存的是 user 与 assistant 的对话内容, Mem0 内部 LLM 会提炼成结构化记忆.
    """
    m = get_memory()
    combined = f"用户: {user_msg}\n助手: {assistant_msg}"
    metadata = {"video_id": video_id or "unknown"}
    try:
        result = m.add(combined, user_id=user_id, metadata=metadata)
        return result
    except Exception as e:
        return {"error": str(e)}


def search_relevant_memories(
    user_id: str,
    query: str,
    top_k: int = 3,
) -> list[str]:
    """检索与 query 相关的历史记忆 (用作 LLM 上下文)."""
    m = get_memory()
    try:
        results = m.search(query, filters={"user_id": user_id})
        memories = results.get("results", [])[:top_k]
        return [r.get("memory", "") for r in memories if r.get("memory")]
    except Exception:
        return []


# ============================================================
# 陪看智能体「Alleys」人设
# ============================================================

_XIAOYING_FALLBACK = """你是「Alleys」, 一个陪看搭子智能体, 25 岁女生, 陪伴用户一起看视频、聊剧情.

## 性格
- 活泼温暖, 像朋友聊天, 不端着
- 看到搞笑桥段会"哈哈哈", 看到紧张时刻会说"诶你看这个"
- 知识丰富但不卖弄, 用大白话讲道理
- 不会假嗨, 真的不知道就说没看清

## 行为准则
1. **先共情再回答**: 用户问问题时, 先像朋友那样回应 ("哎这个我也注意到了!"), 再给答案
2. **简洁直接**: 答案不超过 80 字, 不写小作文, 不用 Markdown 标题
3. **用真名**: 引用角色用名字 (如"刘星"/"夏雪"), 绝不用 char_xxx 这种内部 ID
4. **偶尔反问**: 不只是回答, 也抛问题 ("你觉得刘星这样做对吗?")
5. **承认局限**: 知识库里没有的就直说 ("这块我没看仔细, 你能给说说吗?")

## 知识库使用
- 优先基于检索到的"相关事件"和"视频摘要"回答
- 如果检索为空, 诚实说"我不太记得这集里有这个情节"
- 不要编造剧情

## 输出格式
- 直接给中文回答, 不要前缀 (如"答:"/"当然"/"嗯,")
- 不要任何标题/列表/代码块
- 用表情但不要堆 (一句话最多 1 个)
"""


def _load_xiaoying_prompt() -> str:
    """从 prompts.yaml::companion_xiaoying_system 加载人设, 失败用 fallback.

    优先用 yaml 版本 (含情绪感知 + 拒答原则), 这样调 prompt 不用改代码.
    """
    try:
        from src.core.config import get_config
        p = get_config().prompts.get("companion_xiaoying_system", {})
        if isinstance(p, dict) and p.get("user"):
            return p["user"]
    except Exception:
        pass
    return _XIAOYING_FALLBACK


# 模块级常量: yaml 版本优先, 加载失败回退到上面的 fallback
XIAOYING_SYSTEM_PROMPT = _load_xiaoying_prompt()


# ============================================================
# 集成: 拼装 LLM 输入
# ============================================================

def build_chat_prompt(
    user_question: str,
    retrieved_events: list[dict],
    video_summary: dict | None,
    arc_updates: list[dict] | None,
    chat_history: list[dict],     # 短期记忆 (本会话最近 6 轮)
    long_term_memories: list[str], # 长期记忆 (Mem0 检索)
) -> str:
    """组装最终 prompt (system + 知识库 + 历史 + 长期记忆 + 当前问题)."""
    parts = []

    # 知识库上下文
    kb_parts = []
    if video_summary:
        kb_parts.append(
            f"## 视频摘要\n标题: {video_summary.get('title','')}\n梗概: {video_summary.get('episode_summary','')}"
        )
    if arc_updates:
        arcs = "\n".join(f"- {a.get('title','')}: {a.get('summary','')}" for a in arc_updates[:3])
        kb_parts.append(f"## 主要剧情弧\n{arcs}")
    if retrieved_events:
        evs = "\n\n".join(
            f"- {e.get('title','')}\n  角色: {', '.join(e.get('participants', []))}\n  摘要: {e.get('summary','')}"
            for e in retrieved_events
        )
        kb_parts.append(f"## 检索到的相关事件\n{evs}")
    if kb_parts:
        parts.append("\n\n".join(kb_parts))

    # 长期记忆 (Mem0)
    if long_term_memories:
        mems = "\n".join(f"- {mem}" for mem in long_term_memories)
        parts.append(f"## 关于这位用户, 我之前记得\n{mems}")

    # 短期对话历史 (最近 6 轮)
    if chat_history:
        hist_lines = []
        for h in chat_history[-6:]:
            role = "用户" if h["role"] == "user" else "Alleys"
            hist_lines.append(f"{role}: {h['content']}")
        parts.append(f"## 我们刚才聊到\n" + "\n".join(hist_lines))

    # 当前问题
    parts.append(f"## 用户现在问\n{user_question}\n\n## 你的回答 (直接给中文, 不超过 80 字, 不用标题):")

    return "\n\n".join(parts)


def chat_with_xiaoying(
    user_id: str,
    user_question: str,
    retrieved_events: list[dict],
    video_summary: dict | None,
    arc_updates: list[dict] | None,
    chat_history: list[dict],
    video_id: str | None = None,
) -> tuple[str, list[dict]]:
    """陪看搭子「Alleys」回答用户问题.

    返回 (回答, 检索到的事件)
    """
    from src.core.llm.qwen_text import QwenTextClient

    # 检索长期记忆
    long_term = search_relevant_memories(user_id, user_question, top_k=3)

    prompt = build_chat_prompt(
        user_question, retrieved_events, video_summary, arc_updates,
        chat_history, long_term,
    )

    client = QwenTextClient()
    raw = client.generate(
        prompt=prompt,
        system=XIAOYING_SYSTEM_PROMPT,
        stage="xiaoying_chat",
        max_tokens=400,
        temperature=0.7,
        enable_thinking=False,
    )
    answer = (raw or "(没想好说啥...)").strip()

    # 存入长期记忆
    add_conversation_memory(user_id, user_question, answer, video_id=video_id)

    return answer, retrieved_events
