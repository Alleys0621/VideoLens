"""VideoLens 陪看智能体「小影」 - 视频陪伴 + 智能问答 + 长期记忆

启动:
  .venv/Scripts/streamlit run scripts/frontend_app.py --server.port 8501
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st
from src.core.config import get_config
from src.core.helpers.json_utils import load_json
from src.eval.stage3_retrieval import BM25Index, build_searchable_text

cfg = get_config()


# ============================================================
# CSS: 现代化设计 (参考 ChatGPT/豆包/通义 2026 风格)
# ============================================================
CUSTOM_CSS = """
<style>
:root {
    --bg-primary: #fafaf8;
    --bg-secondary: #ffffff;
    --bg-tertiary: #f5f3ef;
    --text-primary: #1a1a1a;
    --text-secondary: #6b6b6b;
    --text-tertiary: #a3a3a3;
    --accent: #ff6b35;
    --accent-soft: #fff4ee;
    --border: rgba(0,0,0,0.06);
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
    --shadow-md: 0 8px 24px rgba(0,0,0,0.06);
    --shadow-lg: 0 16px 48px rgba(0,0,0,0.08);
}

.stApp {
    background: var(--bg-primary);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
    color: var(--text-primary);
    background-image:
        radial-gradient(at 0% 0%, rgba(255,107,53,0.04) 0px, transparent 50%),
        radial-gradient(at 95% 95%, rgba(91,142,255,0.03) 0px, transparent 50%);
    background-attachment: fixed;
}

/* 隐藏 streamlit 默认元素 */
#MainMenu, footer, header[data-testid="stHeader"],
.stDeployButton, .stToolbar, [data-testid="stLogo"] {
    display: none !important;
}

/* 主容器 padding */
.stApp > section[data-testid="stMain"] {
    padding: 1rem 1.5rem !important;
    max-width: 100% !important;
}

/* 顶栏 */
.top-bar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.4rem 0 1rem 0;
    margin-bottom: 1rem;
}
.top-bar .brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    color: var(--text-primary);
}
.top-bar .brand-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #ff6b35 0%, #ff9068 100%);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1rem;
    box-shadow: var(--shadow-sm);
}
.top-bar .brand-subtitle {
    color: var(--text-tertiary);
    font-size: 0.78rem;
    margin-left: 0.3rem;
    font-weight: 400;
}

/* 主布局: 视频左 60%, 聊天右 40% */
.main-grid {
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    gap: 1.5rem;
    height: calc(100vh - 100px);
}

/* 视频卡片 */
.video-section {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
}
.video-wrapper {
    background: #000;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    flex: 1;
    min-height: 0;
}
.video-meta-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.7rem 1rem;
    background: var(--bg-secondary);
    border-radius: 14px;
    box-shadow: var(--shadow-sm);
    font-size: 0.85rem;
    color: var(--text-secondary);
    flex-wrap: wrap;
}
.video-meta-bar .meta-title {
    font-weight: 600;
    color: var(--text-primary);
    margin-right: auto;
    font-size: 0.95rem;
}
.video-meta-bar .tag {
    padding: 0.2rem 0.6rem;
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* 聊天卡片 */
.chat-section {
    background: var(--bg-secondary);
    border-radius: 20px;
    box-shadow: var(--shadow-md);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.chat-header {
    padding: 1rem 1.2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.agent-avatar {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #ffb38a 0%, #ff6b35 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(255,107,53,0.3);
}
.agent-info .name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
}
.agent-info .status {
    font-size: 0.72rem;
    color: #4ade80;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}
.agent-info .status::before {
    content: "";
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 8px #4ade80;
}

/* 对话区 */
.chat-body {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
}

/* 推荐问题 chip */
.suggestions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.suggestion-card {
    padding: 0.7rem 1rem;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.suggestion-card:hover {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
    transform: translateX(2px);
}

/* 空状态欢迎 */
.welcome {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--text-tertiary);
}
.welcome-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

/* 重写 streamlit chat_message */
[data-testid="stChatMessage"] {
    padding: 0.6rem 0.9rem !important;
    margin-bottom: 0 !important;
    border-radius: 14px !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
    width: 28px !important;
    height: 28px !important;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    padding: 0.7rem 1rem !important;
    border-radius: 14px !important;
}

/* 输入框 */
[data-testid="stChatInput"] {
    padding: 0.8rem 1rem !important;
    border-top: 1px solid var(--border);
    background: var(--bg-secondary);
}
[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.92rem !important;
    background: var(--bg-tertiary) !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #ff6b35 0%, #ff9068 100%) !important;
    border-radius: 12px !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(255,107,53,0.3) !important;
}

/* 引用展开 */
details {
    background: var(--bg-tertiary);
    border-radius: 10px !important;
    padding: 0.4rem 0.8rem !important;
    margin-top: 0.4rem !important;
    border: 1px solid var(--border) !important;
}
summary {
    color: var(--accent) !important;
    font-weight: 500;
    font-size: 0.78rem !important;
    cursor: pointer;
    list-style: none;
}
summary::before {
    content: "📎 ";
}

/* 下拉框美化 */
[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border-color: var(--border) !important;
    background: var(--bg-secondary) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }
</style>
"""


# ============================================================
# 数据 + 检索
# ============================================================

def _list_videos_with_kb() -> list[str]:
    """列出 视频文件存在 + 已建库 的视频目录."""
    if not os.path.isdir(cfg.output_root):
        return []
    found = []
    for root, dirs, files in os.walk(cfg.output_root):
        if "_global" in dirs: dirs.remove("_global")
        if "_batch_reports" in dirs: dirs.remove("_batch_reports")
        if "stage3_kb.json" not in files: continue
        rel = os.path.relpath(root, cfg.output_root).replace("\\", "/")
        if _find_video_file(rel):
            found.append(rel)
    return sorted(found)


def _find_video_file(video_dir: str) -> str | None:
    videos_root = os.path.join(cfg.data_root, "videos")
    for ext in (".mp4", ".mkv", ".mov", ".avi"):
        for candidate in [
            os.path.join(videos_root, video_dir + ext),
            os.path.join(videos_root, video_dir, os.path.basename(video_dir) + ext),
        ]:
            if os.path.isfile(candidate): return candidate
        for sub in os.listdir(videos_root):
            sub_path = os.path.join(videos_root, sub)
            if os.path.isdir(sub_path):
                for candidate in [
                    os.path.join(sub_path, video_dir + ext),
                    os.path.join(sub_path, video_dir, os.path.basename(video_dir) + ext),
                ]:
                    if os.path.isfile(candidate): return candidate
    return None


def _build_index(events):
    if not events: return None
    return BM25Index([build_searchable_text(e) for e in events])


def _retrieve_events(index, events, query, top_k=5):
    return [events[i] for i, _ in index.search(query, top_k=top_k)]


# ============================================================
# 角色名归一
# ============================================================

_PINYIN_MAP = {
    "xue": "夏雪", "mei": "刘梅", "xing": "刘星", "yu": "夏雨", "donghai": "夏东海",
    "liumei": "刘梅", "liu_mei": "刘梅", "lumei": "刘梅",
    "liu_xing": "刘星", "liuxing": "刘星",
    "xiaxue": "夏雪", "xia_xue": "夏雪",
    "xiayu": "夏雨", "xia_yu": "夏雨",
    "xiadonghai": "夏东海", "xia_donghai": "夏东海", "xiaodonghai": "夏东海",
    "grandma": "奶奶", "grandpa": "爷爷", "mother": "母亲", "father": "父亲",
}
_UNKNOWN_PREFIXES = ("char_unknown", "char_passerby", "char_luren", "char_lu_ren", "char_new", "路人", "新角色")


def _build_char_map() -> dict[str, str]:
    p = os.path.join(cfg.output_root, "_global", "characters.json")
    if not os.path.isfile(p): return {}
    chars = load_json(p)
    m = {}
    for c in chars:
        name = c.get("name", "")
        if not name: continue
        m[c.get("character_id", "")] = name
        for alias in c.get("aliases", []) or []:
            m[alias] = name
        m[name] = name
    return m


def _normalize_participant(p: str, char_map: dict[str, str]) -> str:
    if not p: return ""
    p = p.strip()
    if p in char_map: return char_map[p]
    for prefix in _UNKNOWN_PREFIXES:
        if p.startswith(prefix) or p == prefix: return "未知角色"
    if p.startswith("char_"):
        key = p[5:].lower()
        if key in _PINYIN_MAP: return _PINYIN_MAP[key]
        last = key.rsplit("_", 1)[-1]
        if last in _PINYIN_MAP: return _PINYIN_MAP[last]
        no_us = key.replace("_", "")
        if no_us in _PINYIN_MAP: return _PINYIN_MAP[no_us]
        return "未知角色"
    return p


def _normalize_events(events: list[dict], char_map: dict[str, str]) -> list[dict]:
    new = []
    for e in events:
        ne = dict(e)
        ne["participants"] = [_normalize_participant(p, char_map) for p in (e.get("participants") or [])]
        new_actions = []
        for a in e.get("actions", []) or []:
            na = dict(a)
            t = a.get("target")
            if isinstance(t, str):
                na["target"] = _normalize_participant(t, char_map)
            elif isinstance(t, list):
                na["target"] = [_normalize_participant(x, char_map) for x in t]
            new_actions.append(na)
        ne["actions"] = new_actions
        new.append(ne)
    return new


def _suggested_questions(video_summary: dict, events: list[dict]) -> list[str]:
    """根据视频内容动态生成推荐问题."""
    qs = []
    chars = video_summary.get("character_refs", [])[:2]
    # 取出现频次最高的角色
    from collections import Counter
    char_count = Counter()
    for e in events:
        for p in e.get("participants", []) or []:
            char_count[p] += 1
    top_chars = [c for c, _ in char_count.most_common(3) if c != "未知角色"]

    if top_chars:
        qs.append(f"{top_chars[0]}在这集里有什么故事?")
        if len(top_chars) > 1:
            qs.append(f"{top_chars[0]}和{top_chars[1]}有什么互动?")
    qs.append("这集讲了一个什么故事?")
    qs.append("最有意思的桥段是什么?")
    return qs[:4]


# ============================================================
# Streamlit
# ============================================================

st.set_page_config(page_title="小影陪你看剧 · VideoLens", page_icon="🎬", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session state
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"  # 每个浏览器一个匿名 id
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "kb_cache" not in st.session_state:
    st.session_state.kb_cache = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "char_map_cache" not in st.session_state:
    st.session_state.char_map_cache = _build_char_map()

# === Top Bar ===
st.markdown("""
<div class="top-bar">
  <div class="brand">
    <div class="brand-icon">🎬</div>
    <span>VideoLens<span class="brand-subtitle">小影陪你看剧</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

videos = _list_videos_with_kb()
if not videos:
    st.warning("未找到已建库的视频. 先跑 `python -m scripts.stage3_p345 --video <video_dir>`")
    st.stop()

# 视频选择
top_col1, top_col2 = st.columns([4, 2])
with top_col2:
    selected = st.selectbox("选集", options=videos, label_visibility="collapsed")
    if selected != st.session_state.selected_video:
        st.session_state.selected_video = selected
        st.session_state.chat_history = []  # 切换视频清空对话

# 加载 KB
if selected in st.session_state.kb_cache:
    kb = st.session_state.kb_cache[selected]
else:
    kb_path = os.path.join(cfg.output_root, selected, "stage3_kb.json")
    kb = load_json(kb_path) if os.path.isfile(kb_path) else {}
    st.session_state.kb_cache[selected] = kb

video_summary = kb.get("video_summary", {})
events = _normalize_events(kb.get("events", []), st.session_state.char_map_cache)
arc_updates = kb.get("arc_updates", [])

# === 主布局 ===
left_col, right_col = st.columns([1.55, 1], gap="medium")

# === 左侧: 视频 ===
with left_col:
    st.markdown('<div class="video-section">', unsafe_allow_html=True)

    video_file = _find_video_file(selected)
    if video_file and os.path.isfile(video_file):
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
    else:
        st.error(f"⚠️ 视频未找到: data/videos/{selected}.mp4")

    # 视频元数据
    title = video_summary.get("title", "")
    tone = video_summary.get("tone", "")
    meta_html = f"""
    <div class="video-meta-bar">
      <span class="meta-title">📖 {title or selected}</span>
      {f'<span class="tag">🎭 {tone}</span>' if tone else ''}
      <span class="tag">🎞️ {len(events)} 场</span>
      <span class="tag">🎬 {len(arc_updates)} 弧</span>
    </div>
    """
    st.markdown(meta_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# === 右侧: 聊天 ===
with right_col:
    # Chat header
    st.markdown("""
    <div class="chat-header">
      <div class="agent-avatar">✨</div>
      <div class="agent-info">
        <div class="name">小影</div>
        <div class="status">在线 · 刚看完这集</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 对话区
    chat_container = st.container(height=520)
    with chat_container:
        if not st.session_state.chat_history:
            # 空状态欢迎
            st.markdown(
                f'<div class="welcome">'
                f'<div class="welcome-icon">👋</div>'
                f'<div>我是 <b>小影</b>, 刚陪你看了这集</div>'
                f'<div style="font-size:0.8rem;margin-top:0.4rem">问点啥都行, 或者点下面的推荐 👇</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            # 推荐问题
            st.markdown('<div class="suggestions">', unsafe_allow_html=True)
            suggestions = _suggested_questions(video_summary, events)
            for q in suggestions:
                if st.button(q, key=f"sugg_{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # 功能 1: 关键帧定位 (展示 evidence 对应的帧)
                    if msg.get("keyframes"):
                        for kf in msg["keyframes"]:
                            if os.path.isfile(kf):
                                st.image(kf, use_container_width=True)
                    # 功能 2: 推理过程可视化
                    if msg.get("reasoning"):
                        r = msg["reasoning"]
                        labels = {"kb": "知识库检索", "chitchat": "闲聊", "refuse": "拒答"}
                        label = labels.get(r["intent"], r["intent"])
                        with st.expander(f"推理过程 · {label}"):
                            if r["intent"] == "kb":
                                st.markdown(f"BM25 检索命中 (top score **{r['top_score']}** >= 阈值 {r['threshold']})")
                                st.markdown("**检索到的相关事件**:")
                                for ev in r["retrieved"][:3]:
                                    st.markdown(f"- {ev['title']} (score: {ev['score']})")
                            elif r["intent"] == "refuse":
                                st.markdown(f"知识库里没找到相关情节 (top score {r['top_score']} < {r['threshold']}), 诚实拒答不编造")
                            else:
                                st.markdown("识别为闲聊, 不查知识库直接回应")
                    # 参考事件详情 (KB 模式补充)
                    if msg.get("refs") and msg.get("reasoning", {}).get("intent") == "kb":
                        with st.expander(f"参考事件 ({len(msg['refs'])} 条)"):
                            for ref in msg["refs"]:
                                participants = ", ".join(ref.get("participants", [])[:3])
                                st.markdown(f"**{ref.get('title','')}** · {participants}")
                                st.caption(ref.get("summary", "")[:180])

# === 输入 ===
user_q = st.chat_input("和小影聊聊这集...")

if "pending_question" in st.session_state:
    user_q = st.session_state.pending_question
    del st.session_state.pending_question

if user_q:
    st.session_state.chat_history.append({"role": "user", "content": user_q})

    with st.spinner("小影正在想..."):
        from src.agent.companion import companion_chat
        result = companion_chat(
            query=user_q,
            video_dir=selected,
            user_id=st.session_state.user_id,
            chat_history=st.session_state.chat_history[:-1],
        )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "refs": result["reasoning"]["selected"],   # 兼容旧 expander 展示
        "reasoning": result["reasoning"],          # 推理链
        "keyframes": result["keyframes"],          # 帧定位
    })
    st.rerun()
