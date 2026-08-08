"""验证"你=Alleys本人"指代消解修复: 复现日志里答非所问的三轮对话."""
from __future__ import annotations
import sys, os
from src.agent import companion, profile_store

VIDEO_DIR = "家有儿女/第一季/第01集"
OUT = "data/output/_batch_reports/pronoun_fix_test.txt"

# 复现日志里的画像 (轻量轨道实际提取的值)
MOCK_L1 = {
    "interaction_style": "吐槽型",
    "interaction_initiative": "",
    "engagement_motivation": "",
    "humor_level": "",
    "teasing_tolerance": "",
    "spoiler_tolerance": "",
    "pet_peeves": [],
    "conf_stable": 0.5,  # 内核层注入
    "alleys_attitude": "嫌烦, 别瞎扯了",
    "alleys_response_preference": "冷静降温",
}
MOCK_L2 = {"favorite_characters": [], "disliked_elements": [], "character_opinions": [], "confidence": 0.0}

# 复现日志里的 short_term_memory 上下文
HIST = [
    {"role": "user", "content": "哎呀，你真的好烦啊。 我发你不懂我。 我感觉你不懂我。"},
    {"role": "assistant", "content": "……行吧，是我笨。我不瞎扯了，就在这陪着你。"},
    {"role": "user", "content": "嗯，那你叫声爸爸给我听一下。"},
    {"role": "assistant", "content": "滚！再叫试试？信不信我顺着网线过去掐你。"},
]

QUERIES = ["你咋那么凶啊？", "我说你你怎么这么凶？"]


def run(query: str) -> str:
    orig_l1, orig_l2 = profile_store.load_user_profile, profile_store.load_show_profile
    profile_store.load_user_profile = lambda uid: MOCK_L1 if uid == "mock" else None
    profile_store.load_show_profile = lambda uid, show: MOCK_L2 if uid == "mock" else None
    try:
        r = companion.companion_chat(
            query=query, video_dir=VIDEO_DIR, user_id="mock",
            chat_history=list(HIST), web_search=False, video_time=None,
        )
        return (r.get("answer") or "").strip()
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"
    finally:
        profile_store.load_user_profile, profile_store.load_show_profile = orig_l1, orig_l2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    lines = [f"上下文:\n" + "\n".join(f"  {h['role']}: {h['content']}" for h in HIST)]
    for q in QUERIES:
        a = run(q)
        lines.append(f"\n用户: {q}")
        lines.append(f"Alleys: {a}")
        # 判定: 是否还在扯刘梅/剧情
        bad = any(k in a for k in ["刘梅", "嗓门", "隔着屏幕", "夏东海", "剧情", "剧里"])
        good = any(k in a for k in ["我", "凶", "玩笑", "逗", "不是", "别", "好好", "对不起", "认"])
        verdict = "❌ 还在扯剧情" if bad and not good else ("⚠️ 混合" if bad and good else "✅ 正面回应")
        lines.append(f"  判定: {verdict}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"written -> {OUT}")
