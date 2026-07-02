"""Stage 3: 结构化知识库 (P1-P6)

分层:
  P1: Action 抽取 — 11 类 Communicative Action + evidence (绑定 keyframe + segment_id)
  P2: Event 聚合 — 同集按时间序 Action → Event (含 motivation / outcome / retrieval_text)
  P3: PlotArc 匹配 — 本集 events 匹配到 global_arcs, 产出 arc_updates
  P4: Video 摘要 — 整集梗概 + character_refs + main_arcs
  P5: Global — 跨集角色归一 + arc 终态 (累积到 data/output/_global/)
  P6: 角色深度画像 — 性格深层 + 行为模式带 event/action 例证 (独立跑, 不在 run_stage3)

入口:
  - run_stage3(video_dir): P1→P5 完整流程 (供 orchestrator.run_pipeline)
  - run_p1p2(video_dir):   仅 P1+P2 (供 scripts/stage3_p1p2)
  - run_p345(video_dir):   仅 P3+P4+P5 (供 scripts/stage3_p345)
  - run_p6():              角色深度画像 (供 scripts/stage3_p6, 消费所有集数据)
"""

from src.pipeline.stage3.p1_p2_actions import run_p1p2
from src.pipeline.stage3.p345_kb import run_p345
from src.pipeline.stage3.p6_profile import run_p6


def run_stage3(video_dir: str, output_dir: str | None = None) -> dict:
    """Stage 3 完整流程: P1+P2 → P3+P4+P5.

    供 orchestrator.run_pipeline 调用。需先完成 Stage 1/2 (audio.json + visual.json
    已落到 data/output/{video_dir}/)。

    Returns:
        完整 KB dict (含 events / actions / arc_updates / video_summary / cost).
    """
    run_p1p2(video_dir, save=True)
    return run_p345(video_dir)
