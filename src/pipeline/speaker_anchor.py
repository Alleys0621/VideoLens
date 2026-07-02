"""
基于声纹时间戳的关键帧锚点生成器 (Speaker-Anchor Keyframe Selector)

设计动机见 docs/research notes:
  - 已有 Stage 1 的声纹 segment (speaker_pred + 时间戳), 可零成本产出语义对齐的关键帧锚点
  - TransNetV2 在情景剧 (固定机位 + 长对话镜头) 上过分割, 且完全不利用音频信息
  - 文献路线: Audio-Guided Keyframe Selection (Iyer 2024), Dialogue-Aligned Sampling (Fu ACL 2023)

锚点类型:
  - "midpoint": 每个说话段的中点 — 保证帧对应一段完整台词
  - "switch":   说话人切换点 + 偏移 — 捕获反应镜头 / reaction shot
  - "silence":  无对话静默段 (>min_silence_s) 的兜底锚点, 可选触发 SBD

纯函数模块, 不读视频, 不写文件, 便于测试与复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

AnchorType = Literal["midpoint", "switch", "silence"]


@dataclass
class Anchor:
    """单个关键帧锚点"""
    timestamp: float                 # 视频时间戳 (秒)
    anchor_type: AnchorType
    speaker: str                     # 对应 speaker_pred, 静默段为 ""
    segment_id: str | None           # 关联的 audio segment id, 静默段为 None
    text: str = ""                   # 关联台词 (仅 midpoint)
    switch_from: str | None = None   # 仅 switch: 前一个说话人
    switch_to: str | None = None     # 仅 switch: 后一个说话人

    def __repr__(self) -> str:
        return (
            f"Anchor(t={self.timestamp:.2f}s, {self.anchor_type}, "
            f"spk={self.speaker or '-'})"
        )


def _segment_midpoint(seg: dict) -> float:
    """单段中点"""
    return 0.5 * (float(seg["begin_time"]) + float(seg["end_time"]))


def _has_speaker(seg: dict) -> bool:
    """该段是否有有效 speaker_pred (非空且非纯空白)"""
    sp = seg.get("speaker_pred", "") or ""
    return bool(sp.strip())


def generate_midpoint_anchors(
    segments: Iterable[dict],
    min_duration: float = 0.3,
    require_speaker: bool = True,
) -> list[Anchor]:
    """为每段对话生成一个中点锚点.

    Args:
        segments: Stage 1 audio.json 中的 segments 列表
        min_duration: 短于此长度的段跳过 (噪声/单字)
        require_speaker: 是否要求 speaker_pred 非空; False 时也保留匿名段

    Returns:
        锚点列表, 按时间戳升序
    """
    anchors: list[Anchor] = []
    for seg in segments:
        begin = float(seg["begin_time"])
        end = float(seg["end_time"])
        if end - begin < min_duration:
            continue
        if require_speaker and not _has_speaker(seg):
            continue
        anchors.append(
            Anchor(
                timestamp=_segment_midpoint(seg),
                anchor_type="midpoint",
                speaker=(seg.get("speaker_pred", "") or "").strip(),
                segment_id=seg.get("segment_id"),
                text=(seg.get("text", "") or "").strip(),
            )
        )
    anchors.sort(key=lambda a: a.timestamp)
    return anchors


def generate_switch_anchors(
    segments: Iterable[dict],
    offset: float = 0.3,
    skip_same_speaker: bool = True,
) -> list[Anchor]:
    """在说话人切换点附近生成辅助锚点, 用于捕获 reaction shot.

    Args:
        segments: Stage 1 segments
        offset: 切换点向后偏移的秒数 (反应镜头通常出现在新说话人开口后)
        skip_same_speaker: 相邻段若是同一 speaker_pred 则不生成切换锚点

    Returns:
        锚点列表
    """
    segs = list(segments)
    anchors: list[Anchor] = []
    for i in range(1, len(segs)):
        prev, cur = segs[i - 1], segs[i]
        sp_prev = (prev.get("speaker_pred", "") or "").strip()
        sp_cur = (cur.get("speaker_pred", "") or "").strip()
        if not sp_cur:
            continue
        if skip_same_speaker and sp_prev == sp_cur and sp_prev:
            continue
        switch_t = float(cur["begin_time"])
        anchors.append(
            Anchor(
                timestamp=switch_t + offset,
                anchor_type="switch",
                speaker=sp_cur,
                segment_id=cur.get("segment_id"),
                switch_from=sp_prev or None,
                switch_to=sp_cur,
            )
        )
    anchors.sort(key=lambda a: a.timestamp)
    return anchors


def generate_silence_anchors(
    segments: Iterable[dict],
    content_end: float,
    min_silence: float = 3.0,
    content_start: float = 0.0,
) -> list[Anchor]:
    """在无对话静默段生成兜底锚点.

    仅对 >min_silence 的静默段产锚点, 后续可触发 SBD.
    每个静默段一个中点锚点.

    Returns:
        锚点列表
    """
    segs = sorted(list(segments), key=lambda s: float(s["begin_time"]))
    anchors: list[Anchor] = []
    # 构造静默区间: [content_start, seg0.begin], [seg_i.end, seg_{i+1}.begin], ..., [segN.end, content_end]
    gaps: list[tuple[float, float]] = []
    if not segs:
        if content_end - content_start >= min_silence:
            gaps.append((content_start, content_end))
    else:
        first_begin = float(segs[0]["begin_time"])
        if first_begin - content_start >= min_silence:
            gaps.append((content_start, first_begin))
        for i in range(1, len(segs)):
            prev_end = float(segs[i - 1]["end_time"])
            cur_begin = float(segs[i]["begin_time"])
            if cur_begin - prev_end >= min_silence:
                gaps.append((prev_end, cur_begin))
        last_end = float(segs[-1]["end_time"])
        if content_end - last_end >= min_silence:
            gaps.append((last_end, content_end))

    for g_start, g_end in gaps:
        anchors.append(
            Anchor(
                timestamp=0.5 * (g_start + g_end),
                anchor_type="silence",
                speaker="",
                segment_id=None,
            )
        )
    return anchors


def dedupe_anchors(
    anchors: Iterable[Anchor],
    min_gap_s: float = 0.5,
) -> list[Anchor]:
    """合并过近的锚点. 优先级 midpoint > switch > silence.

    Args:
        min_gap_s: 任意两个保留锚点之间的最小时间间隔

    Returns:
        去重后的锚点列表 (按时间升序)
    """
    priority = {"midpoint": 0, "switch": 1, "silence": 2}
    sorted_anchors = sorted(anchors, key=lambda a: (a.timestamp, priority[a.anchor_type]))
    kept: list[Anchor] = []
    for a in sorted_anchors:
        if kept and a.timestamp - kept[-1].timestamp < min_gap_s:
            # 距离上一个太近, 仅当当前优先级更高才替换
            if priority[a.anchor_type] < priority[kept[-1].anchor_type]:
                kept[-1] = a
            continue
        kept.append(a)
    kept.sort(key=lambda a: a.timestamp)
    return kept


def build_anchors(
    segments: Iterable[dict],
    content_range: dict | None = None,
    *,
    enable_switch: bool = True,
    enable_silence: bool = True,
    switch_offset: float = 0.3,
    min_silence: float = 3.0,
    min_segment_duration: float = 0.3,
    dedupe_gap: float = 0.5,
) -> list[Anchor]:
    """一站式构造所有锚点 + 去重.

    Args:
        segments: Stage 1 audio.json 的 segments
        content_range: {"start": float, "end": float}; 用于 silence 兜底
        enable_switch / enable_silence: 开关
        switch_offset: 切换锚点偏移
        min_silence: 静默段最小长度
        min_segment_duration: midpoint 锚点的最小段长
        dedupe_gap: 去重最小间隔

    Returns:
        最终锚点列表
    """
    seg_list = list(segments)
    anchors: list[Anchor] = []
    anchors.extend(
        generate_midpoint_anchors(seg_list, min_duration=min_segment_duration)
    )
    if enable_switch:
        anchors.extend(generate_switch_anchors(seg_list, offset=switch_offset))
    if enable_silence and content_range:
        anchors.extend(
            generate_silence_anchors(
                seg_list,
                content_end=float(content_range["end"]),
                content_start=float(content_range.get("start", 0.0)),
                min_silence=min_silence,
            )
        )
    return dedupe_anchors(anchors, min_gap_s=dedupe_gap)

