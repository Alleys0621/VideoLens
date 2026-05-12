"""单阶段测试服务"""

import os
import time

from vl.core.config import get_config
from vl.core.models.scene import Scene
from vl.core.helpers.json_utils import load_json, save_json
from vl.core.helpers.ffmpeg import extract_audio
from vl.core.logging import get_logger

logger = get_logger()


def load_stage_prerequisites(video_id: str, stage: int) -> dict:
    """为 test-stage 加载前置数据。

    Returns:
        包含 stage 所需前置数据的 dict
    """
    config = get_config()
    output_dir = config.output_root

    def _check(path, label):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} 不存在: {path}\n请先运行 Stage {stage - 1} 或更早的阶段。")
        return path

    if stage == 1:
        video_path = os.path.join(config.data_root, "videos", f"{video_id}.mp4")
        _check(video_path, "视频文件")
        return {"video_path": video_path}

    scenes_json = os.path.join(output_dir, "stage1_scenes", video_id, "scenes.json")
    _check(scenes_json, "场景数据")

    if stage == 2:
        audio_path = os.path.join(output_dir, "stage2_features", "preprocessing", f"{video_id}.wav")
        if not os.path.isfile(audio_path):
            logger.info(f"音频文件不存在，将从视频提取: {audio_path}")
        return {"scenes_json": scenes_json, "audio_path": audio_path}

    # Stage 3/4/5: 需要 scenes + transcripts + characters
    scenes_data = load_json(scenes_json)
    scenes = [Scene.from_dict(s) for s in scenes_data]

    metadata_path = os.path.join(output_dir, "stage1_scenes", video_id, "metadata.json")
    if os.path.isfile(metadata_path):
        metadata = load_json(metadata_path)
        scenes = [Scene.from_dict(s) for s in metadata.get("scenes", [])]

    # 构建 scene_transcripts
    transcript_path = os.path.join(output_dir, "stage2_features", video_id, "transcript.json")
    scene_transcripts = {}
    if os.path.isfile(transcript_path):
        segments = load_json(transcript_path)
        for seg in segments:
            sid = seg.get("scene_id", "")
            text = seg.get("text", "")
            if sid and text:
                scene_transcripts.setdefault(sid, []).append(text)

    # 加载角色信息
    characters_path = os.path.join(output_dir, "stage2_features", video_id, "characters.json")
    characters_info = ""
    if os.path.isfile(characters_path):
        characters = load_json(characters_path)
        names = [c.get("label", "") for c in characters if c.get("label")]
        characters_info = "、".join(names)

    return {
        "scenes": scenes,
        "scene_transcripts": scene_transcripts,
        "characters_info": characters_info,
    }


def run_test_stage(video_id: str, stage: int) -> str:
    """执行单个阶段测试。

    Returns:
        测试结果描述
    """
    config = get_config()
    output_dir = config.output_root
    video_title = video_id

    from vl.core.paths import PathManager
    paths = PathManager(video_id)

    prereq = load_stage_prerequisites(video_id, stage)
    t0 = time.time()

    if stage == 1:
        from vl.pipeline.stage1_ingestion import run_stage1
        scenes = run_stage1(
            video_path=prereq["video_path"],
            paths=paths,
            config=config,
        )
        save_json(
            [s.to_dict() for s in scenes],
            paths.scenes_json_path,
        )
        result = f"检测到 {len(scenes)} 个场景"

    elif stage == 2:
        from vl.pipeline.stage2_analysis import run_stage2

        scenes = [Scene.from_dict(s) for s in load_json(prereq["scenes_json"])]

        audio_path = prereq["audio_path"]
        if not os.path.isfile(audio_path):
            logger.info("音频不存在，从视频提取...")
            video_path = os.path.join(config.data_root, "videos", f"{video_id}.mp4")
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            extract_audio(video_path, audio_path)

        scene_transcripts = run_stage2(
            video_path=os.path.join(config.data_root, "videos", f"{video_id}.mp4"),
            video_id=video_id,
            scenes=scenes,
            audio_path=audio_path,
            paths=paths,
            config=config,
        )
        total_segs = sum(len(v) for v in scene_transcripts.values())
        result = f"{len(scene_transcripts)} 个场景有台词, 共 {total_segs} 个转录片段"

    elif stage == 3:
        from vl.pipeline.stage3_understanding import run_stage3
        run_stage3(
            video_id=video_id,
            scenes=prereq["scenes"],
            scene_transcripts=prereq["scene_transcripts"],
            paths=paths,
            config=config,
        )
        captioned = sum(1 for s in prereq["scenes"] if s.structured_caption)
        result = f"{len(prereq['scenes'])} 个场景, {captioned} 个有结构化描述"

    elif stage == 4:
        from vl.pipeline.stage4_event_builder import run_stage4
        events = run_stage4(
            video_id=video_id,
            scenes=prereq["scenes"],
            scene_transcripts=prereq["scene_transcripts"],
            paths=paths,
            config=config,
        )
        result = f"提取 {len(events)} 个事件"

    elif stage == 5:
        from vl.pipeline.stage5_knowledge import run_stage5
        kb = run_stage5(
            video_id=video_id,
            video_title=video_title,
            scenes=prereq["scenes"],
            scene_transcripts=prereq["scene_transcripts"],
            paths=paths,
            config=config,
        )
        events_count = len(kb.get(video_title, {}))
        sub_count = sum(len(v) for v in kb.get(video_title, {}).values())
        result = f"{events_count} 个大事件, {sub_count} 个小事件"

    else:
        raise ValueError(f"无效的阶段 {stage}，请选择 1-5")

    elapsed = time.time() - t0
    return f"{result}\n耗时: {elapsed:.1f}s"
