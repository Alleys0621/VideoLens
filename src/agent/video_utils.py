"""视频陪看通用工具函数.

唯一入口是 companion_chat; 这里放跨模块复用的数据加载 / 关键帧 / 异步副作用.

依赖:
  - stage3_kb.json / visual.json / audio.json 数据加载
  - keyframe 路径解析
  - Mem0 异步写 + L1/L2 画像异步更新 (threading, 不阻塞回复)
"""

from __future__ import annotations

import functools
import os
import threading

from src.core.config import get_config
from src.core.helpers.json_utils import load_json
from src.core.logging import get_logger
from src.agent.mem0_client import add_conversation_memory
from src.agent.profile_store import (
    increment_message_counter,
    PROFILE_UPDATE_THRESHOLD,
)
from src.agent.profile_updater import maybe_update_user_profile, maybe_update_show_profile

logger = get_logger()


@functools.lru_cache(maxsize=8)
def load_episode_data(video_dir: str) -> tuple[list, list, list, list]:
    """加载一集的 events / actions / scenes / audio_segments.

    内存缓存: 同 video_dir 的多次调用直接返回 (lru_cache, maxsize=8).
    若 stage3_kb.json 被重新生成, 需 `load_episode_data.cache_clear()`.

    Returns:
        (events, actions, scenes, segments) —
        events/actions 来自 stage3_kb.json,
        scenes 来自 visual.json,
        segments 来自 audio.json (Stage 1 ASR, 含 begin_time/end_time/speaker_pred/text).
    """
    cfg = get_config()
    ep_dir = os.path.join(cfg.output_root, video_dir)
    kb_path = os.path.join(ep_dir, "stage3_kb.json")
    visual_path = os.path.join(ep_dir, "visual.json")
    audio_path = os.path.join(ep_dir, "audio.json")

    if not os.path.isfile(kb_path):
        raise FileNotFoundError(f"未建库: {kb_path}; 先跑 Stage 1-3")

    kb = load_json(kb_path)
    events = kb.get("events", []) or []
    actions = kb.get("actions", []) or []

    scenes = []
    if os.path.isfile(visual_path):
        scenes = load_json(visual_path).get("scenes", []) or []

    segments = []
    if os.path.isfile(audio_path):
        segments = load_json(audio_path).get("segments", []) or []

    return events, actions, scenes, segments


def scene_id_to_keyframe(scene_id: str, scenes: list) -> str:
    """scene_id → 主 keyframe 路径."""
    for s in scenes:
        if s.get("scene_id") == scene_id:
            kfps = s.get("keyframe_paths") or []
            if kfps:
                return kfps[0]
    return ""


def events_to_keyframes(events: list[dict], scenes: list, max_frames: int = 3) -> list[str]:
    """从 events 的 evidence.scene_ids 提取关键帧路径 (去重, 限 max_frames)."""
    frames = []
    seen = set()
    for e in events:
        for ev in e.get("evidence", []) or []:
            for sid in ev.get("scene_ids", []) or []:
                if sid in seen:
                    continue
                kf = scene_id_to_keyframe(sid, scenes)
                if kf:  # 路径存在性由 UI 端检查 (避免每次 stat)
                    frames.append(kf)
                    seen.add(sid)
                    if len(frames) >= max_frames:
                        return frames
    return frames


def async_add_memory(user_id: str, query: str, answer: str, video_id: str) -> None:
    """异步写 Mem0 记忆 (线程, 不阻塞返回 — Mem0 写入慢)."""
    def _write():
        try:
            add_conversation_memory(user_id, query, answer, video_id=video_id)
        except Exception as e:
            logger.warning(f"[Mem0] async add 失败: {e}")

    threading.Thread(target=_write, daemon=True).start()


def async_maybe_update_profile(
    user_id: str, chat_history: list[dict], query: str, answer: str, show: str = "",
) -> None:
    """异步累加对话计数, 达阈值触发 L1 + L2 画像增量更新 (不阻塞回复)."""
    def _run():
        try:
            n = increment_message_counter(user_id)
            if n >= PROFILE_UPDATE_THRESHOLD:
                hist = list(chat_history or []) + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ]
                maybe_update_user_profile(user_id, hist)
                if show:
                    maybe_update_show_profile(user_id, show, hist)
        except Exception as e:
            logger.warning(f"[profile] async trigger 失败: {e}")

    threading.Thread(target=_run, daemon=True).start()
