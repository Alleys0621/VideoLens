"""流水线编排器 - 控制各阶段执行和断点续跑"""

import json
import os
import time
from dataclasses import dataclass

from vl.core.config import get_config
from vl.core.paths import PathManager
from vl.core.logging import get_logger
from vl.core.helpers.json_utils import save_json, load_json
from vl.core.helpers.ffmpeg import extract_audio
from vl.core.models.scene import Scene
from vl.core.models.video_meta import VideoMeta

from vl.pipeline.stage1_ingestion import run_stage1
from vl.pipeline.stage2_analysis import run_stage2
from vl.pipeline.stage3_indexing import run_stage3

logger = get_logger()


@dataclass
class Checkpoint:
    """处理断点状态"""
    video_id: str
    stages: dict[str, str]  # stage_name -> "pending" | "running" | "done"
    start_time: float = 0.0
    last_scene_index: int = -1

    def is_done(self, stage: str) -> bool:
        return self.stages.get(stage) == "done"

    def mark(self, stage: str, status: str):
        self.stages[stage] = status

    def save(self, path: str):
        save_json({
            "video_id": self.video_id,
            "stages": self.stages,
            "start_time": self.start_time,
            "last_scene_index": self.last_scene_index,
        }, path)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        data = load_json(path)
        return cls(
            video_id=data["video_id"],
            stages=data.get("stages", {}),
            start_time=data.get("start_time", 0),
            last_scene_index=data.get("last_scene_index", -1),
        )


class PipelineOrchestrator:
    """流水线主控制器"""

    TOTAL_STAGES = 3

    def __init__(self, video_path: str, genre: str = "movie"):
        self.config = get_config()
        self.video_path = video_path
        self.genre = genre
        self.video_id = os.path.splitext(os.path.basename(video_path))[0]
        self.paths = PathManager(self.video_id)
        self.checkpoint = self._load_or_create_checkpoint()
        self._stage_timings: dict[str, float] = {}

    def _load_or_create_checkpoint(self) -> Checkpoint:
        """加载或创建断点"""
        cp_path = self.paths.checkpoint_path
        if os.path.isfile(cp_path):
            logger.info(f"发现断点文件，尝试续跑: {cp_path}")
            return Checkpoint.load(cp_path)
        return Checkpoint(
            video_id=self.video_id,
            stages={
                "stage1": "pending",
                "stage2": "pending",
                "stage3": "pending",
            },
        )

    def run(self, resume: bool = True):
        """运行完整流水线"""
        logger.info(f"开始处理视频: {self.video_path}")
        logger.info(f"视频ID: {self.video_id}, 类型: {self.genre}")

        self.checkpoint.start_time = time.time()

        # Stage 1: 场景分割
        if not (resume and self.checkpoint.is_done("stage1")):
            logger.info("═══ Stage 1/%d: 场景分割 ═══", self.TOTAL_STAGES)
            self._run_stage("stage1", self._stage1)
        else:
            logger.info("[Stage 1] 已完成，跳过")

        # Stage 2: 多维分析
        if not (resume and self.checkpoint.is_done("stage2")):
            logger.info("═══ Stage 2/%d: 多维分析 ═══", self.TOTAL_STAGES)
            self._run_stage("stage2", self._stage2)
        else:
            logger.info("[Stage 2] 已完成，跳过")

        # Stage 3: 索引构建
        if not (resume and self.checkpoint.is_done("stage3")):
            logger.info("═══ Stage 3/%d: 索引构建 ═══", self.TOTAL_STAGES)
            self._run_stage("stage3", self._stage3)
        else:
            logger.info("[Stage 3] 已完成，跳过")

        elapsed = time.time() - self.checkpoint.start_time
        logger.info(f"处理完成！总耗时: {elapsed:.1f}秒")

        # 输出各阶段耗时汇总
        if self._stage_timings:
            logger.info("── 各阶段耗时 ──")
            for stage_name, stage_elapsed in self._stage_timings.items():
                logger.info("  %s: %.1f秒", stage_name, stage_elapsed)

    def _run_stage(self, name: str, func):
        """运行单个阶段，自动更新断点"""
        self.checkpoint.mark(name, "running")
        self.checkpoint.save(self.paths.checkpoint_path)
        stage_start = time.time()
        try:
            func()
            stage_elapsed = time.time() - stage_start
            self._stage_timings[name] = stage_elapsed
            self.checkpoint.mark(name, "done")
            self.checkpoint.save(self.paths.checkpoint_path)
            logger.info(f"[{name}] 完成 (耗时: {stage_elapsed:.1f}秒)")
        except Exception as e:
            stage_elapsed = time.time() - stage_start
            self._stage_timings[name] = stage_elapsed
            self.checkpoint.mark(name, "failed")
            self.checkpoint.save(self.paths.checkpoint_path)
            logger.error(f"[{name}] 失败 (耗时: {stage_elapsed:.1f}秒): {e}")
            raise

    def _stage1(self):
        """场景分割"""
        scenes = run_stage1(
            video_path=self.video_path,
            output_dir=self.paths.video_scenes_dir,
            config=self.config,
        )
        # 保存场景列表
        save_json(
            [s.to_dict() for s in scenes],
            self.paths.scenes_json_path,
        )
        logger.info(f"场景分割完成: 共检测到 {len(scenes)} 个场景")

    def _stage2(self):
        """多维分析"""
        # 加载场景
        scenes_data = load_json(self.paths.scenes_json_path)
        scenes = [Scene.from_dict(s) for s in scenes_data]
        logger.info(f"加载了 {len(scenes)} 个场景，准备进行多维分析")

        # 提取音频
        logger.info("提取音频...")
        extract_audio(self.video_path, self.paths.audio_path)

        run_stage2(
            video_path=self.video_path,
            video_id=self.video_id,
            scenes=scenes,
            audio_path=self.paths.audio_path,
            output_dir=self.paths.output_root,
            config=self.config,
        )

    def _stage3(self):
        """索引构建"""
        # 优先从 metadata.json（Stage 2 enriched）加载，回退到 scenes.json
        metadata_path = os.path.join(self.paths.output_root, "scenes", self.video_id, "metadata.json")
        if os.path.isfile(metadata_path):
            metadata = load_json(metadata_path)
            scenes = [Scene.from_dict(s) for s in metadata.get("scenes", [])]
        else:
            scenes_data = load_json(self.paths.scenes_json_path)
            scenes = [Scene.from_dict(s) for s in scenes_data]
        logger.info(f"加载了 {len(scenes)} 个场景，准备构建索引")

        run_stage3(
            video_id=self.video_id,
            scenes=scenes,
            output_dir=self.paths.output_root,
            config=self.config,
        )
