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
from vl.pipeline.stage3_understanding import run_stage3
from vl.pipeline.stage4_alignment import run_stage4
from vl.pipeline.stage5_knowledge import run_stage5

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

    TOTAL_STAGES = 5

    def __init__(self, video_path: str, genre: str = "cartoon"):
        self.config = get_config()
        self.video_path = video_path
        self.genre = genre
        self.video_id = os.path.splitext(os.path.basename(video_path))[0]
        self.video_title = self.video_id  # 默认用文件名，可后续扩展
        self.paths = PathManager(self.video_id)
        self.checkpoint = self._load_or_create_checkpoint()
        self._stage_timings: dict[str, float] = {}
        # Stage 间传递的数据
        self._scene_transcripts: dict[str, list[str]] = {}
        self._characters_info: str = ""

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
                "stage4": "pending",
                "stage5": "pending",
            },
        )

    def run(self, resume: bool = True):
        """运行完整流水线"""
        logger.info(f"开始处理视频: {self.video_path}")
        logger.info(f"视频ID: {self.video_id}, 类型: {self.genre}")

        self.checkpoint.start_time = time.time()

        stages = [
            ("stage1", "场景分割", self._stage1),
            ("stage2", "特征提取", self._stage2),
            ("stage3", "视觉语义理解", self._stage3),
            ("stage4", "多模态时序对齐", self._stage4),
            ("stage5", "结构化知识库", self._stage5),
        ]

        for stage_name, stage_desc, stage_func in stages:
            if resume and self.checkpoint.is_done(stage_name):
                logger.info(f"[Stage {stage_name}] 已完成，跳过")
                # 续跑时需要加载 stage2 的中间数据
                if stage_name == "stage2":
                    self._load_stage2_outputs()
                continue

            logger.info(f"═══ {stage_name}/{self.TOTAL_STAGES}: {stage_desc} ═══")
            self._run_stage(stage_name, stage_func)

        elapsed = time.time() - self.checkpoint.start_time
        logger.info(f"处理完成！总耗时: {elapsed:.1f}秒")

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

    def _load_scenes(self) -> list[Scene]:
        """加载场景列表"""
        scenes_data = load_json(self.paths.scenes_json_path)
        return [Scene.from_dict(s) for s in scenes_data]

    def _load_stage2_outputs(self):
        """续跑时加载 stage2 的输出 (transcripts + characters)"""
        # 加载 transcripts
        transcript_path = os.path.join(
            self.paths.output_root, "transcripts", self.video_id, "transcript.json"
        )
        if os.path.isfile(transcript_path):
            segments = load_json(transcript_path)
            self._scene_transcripts = {}
            for seg in segments:
                sid = seg.get("scene_id", "")
                text = seg.get("text", "")
                if sid and text:
                    self._scene_transcripts.setdefault(sid, []).append(text)

        # 加载 characters
        characters_path = os.path.join(
            self.paths.output_root, "characters", self.video_id, "characters.json"
        )
        if os.path.isfile(characters_path):
            characters = load_json(characters_path)
            names = [c.get("label", "") for c in characters if c.get("label")]
            self._characters_info = "、".join(names)

    def _stage1(self):
        """Stage 1: 场景分割"""
        scenes = run_stage1(
            video_path=self.video_path,
            output_dir=self.paths.video_scenes_dir,
            config=self.config,
        )
        save_json(
            [s.to_dict() for s in scenes],
            self.paths.scenes_json_path,
        )
        logger.info(f"场景分割完成: 共检测到 {len(scenes)} 个场景")

    def _stage2(self):
        """Stage 2: 特征提取 (ASR + CLIP + 人脸)"""
        scenes = self._load_scenes()
        logger.info(f"加载了 {len(scenes)} 个场景，准备进行特征提取")

        # 提取音频
        logger.info("提取音频...")
        extract_audio(self.video_path, self.paths.audio_path)

        scene_transcripts = run_stage2(
            video_path=self.video_path,
            video_id=self.video_id,
            scenes=scenes,
            audio_path=self.paths.audio_path,
            output_dir=self.paths.output_root,
            config=self.config,
        )

        # 保存更新后的场景 (含 content_type 等新增字段)
        save_json(
            [s.to_dict() for s in scenes],
            self.paths.scenes_json_path,
        )

        # 保存给后续 stage 使用
        self._scene_transcripts = scene_transcripts

        # 加载角色信息
        characters_path = os.path.join(
            self.paths.output_root, "characters", self.video_id, "characters.json"
        )
        if os.path.isfile(characters_path):
            characters = load_json(characters_path)
            names = [c.get("label", "") for c in characters if c.get("label")]
            self._characters_info = "、".join(names)

    def _stage3(self):
        """Stage 3: 视觉语义理解 (VLM 结构化描述 + FAISS 索引)"""
        # 优先从 metadata.json 加载 enriched scenes，否则从 scenes.json
        metadata_path = os.path.join(self.paths.output_root, "scenes", self.video_id, "metadata.json")
        if os.path.isfile(metadata_path):
            metadata = load_json(metadata_path)
            scenes = [Scene.from_dict(s) for s in metadata.get("scenes", [])]
        else:
            scenes = self._load_scenes()

        logger.info(f"加载了 {len(scenes)} 个场景，准备视觉语义理解")

        run_stage3(
            video_id=self.video_id,
            video_title=self.video_title,
            genre=self.genre,
            scenes=scenes,
            scene_transcripts=self._scene_transcripts,
            characters_info=self._characters_info,
            output_dir=self.paths.output_root,
            config=self.config,
        )

    def _stage4(self):
        """Stage 4: 多模态时序对齐"""
        # 加载最新场景数据 (含 structured_caption)
        metadata_path = os.path.join(self.paths.output_root, "scenes", self.video_id, "metadata.json")
        if os.path.isfile(metadata_path):
            metadata = load_json(metadata_path)
            scenes = [Scene.from_dict(s) for s in metadata.get("scenes", [])]
        else:
            scenes = self._load_scenes()

        logger.info(f"加载了 {len(scenes)} 个场景，准备多模态对齐")

        self._aligned_timeline = run_stage4(
            video_id=self.video_id,
            video_title=self.video_title,
            scenes=scenes,
            scene_transcripts=self._scene_transcripts,
            output_dir=self.paths.output_root,
            config=self.config,
        )

    def _stage5(self):
        """Stage 5: 结构化知识库生成"""
        # 加载对齐后的时间线
        timeline_path = os.path.join(
            self.paths.output_root, "alignment", self.video_id, "aligned_timeline.json"
        )
        if os.path.isfile(timeline_path):
            aligned_timeline = load_json(timeline_path)
        else:
            logger.error("未找到对齐数据，请先运行 Stage 4")
            raise FileNotFoundError(f"对齐数据不存在: {timeline_path}")

        run_stage5(
            video_id=self.video_id,
            video_title=self.video_title,
            genre=self.genre,
            aligned_timeline=aligned_timeline,
            output_dir=self.paths.output_root,
            config=self.config,
        )
