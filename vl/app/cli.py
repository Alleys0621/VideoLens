"""VideoLens CLI - 命令行接口"""

import json
import os
import sys

import numpy as np

import typer

from vl.core.logging import get_logger

logger = get_logger()

VIDEOLENS_VERSION = "0.1.0"

_BANNER = f"""
  __   __     _       ___      _
  \\ \\ / /__ _| |_ ___| __)_ __| |
   \\ V / _` |  _/ _ \\ __| '__| |
    | | (_| | ||  __/ |_| |  | |
    |_|\\__,_|\\__\\___/\\__|_|  |_|   v{VIDEOLENS_VERSION}
"""

app = typer.Typer(name="videolens", help="VideoLens - 影视视频内容分析与检索")


# ──────────────────────────────────────────────────────────
# 共用工具函数
# ──────────────────────────────────────────────────────────

def _retrieve_scenes(query: str, video: str, top_k: int = 10):
    """CLIP 编码查询 → FAISS 检索相关场景，返回 (results, config)"""
    from vl.core.config import get_config
    from vl.vision.clip_encoder import CLIPEncoder
    from vl.store.vector_store import VectorStore

    config = get_config()
    clip = CLIPEncoder(model_name=config.model_clip)

    # --- 查询扩展 ---
    queries = [query]
    if config.dashscope_api_key:
        try:
            from vl.core.llm.qwen_text import QwenTextClient
            qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
            expand_prompt = config.prompts.get("query_expand", {})
            user_tpl = expand_prompt.get("user", "")
            sys_prompt = expand_prompt.get("system", "")
            if user_tpl:
                expanded = qwen.generate(user_tpl.format(query=query), system=sys_prompt)
                if expanded:
                    extra = [w.strip() for w in expanded.split() if w.strip()]
                    queries.extend(extra)
                    logger.info(f"查询扩展: {query} -> {queries}")
        except Exception as e:
            logger.warning(f"查询扩展失败 (使用原始查询): {e}")

    # 编码查询
    query_vecs = clip.encode_texts(queries)
    query_vec = np.mean(query_vecs, axis=0)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # 查找索引
    index_root = os.path.join(config.output_root, "stage3_captions")
    if not os.path.isdir(index_root):
        return None, config

    all_results = []
    video_dirs = [video] if video else os.listdir(index_root)

    for vid in video_dirs:
        idx_dir = os.path.join(index_root, vid)
        faiss_path = os.path.join(idx_dir, "index.faiss")
        doc_path = os.path.join(idx_dir, "doc_store.json")

        if not os.path.isfile(faiss_path) or not os.path.isfile(doc_path):
            continue

        store = VectorStore(dim=clip.dim)
        store.load(faiss_path)

        results = store.search(query_vec, top_k=top_k)
        docs = json.load(open(doc_path, "r", encoding="utf-8"))
        doc_map = {d["scene_id"]: d for d in docs}

        for scene_id, score in results:
            doc = doc_map.get(scene_id, {})
            all_results.append({
                "video_id": vid,
                "score": score,
                **doc,
            })

    # 排序
    all_results.sort(key=lambda x: -x["score"])
    all_results = all_results[:top_k]

    # LLM 重排
    if config.retrieval_rerank and config.dashscope_api_key and all_results:
        try:
            from vl.core.llm.qwen_text import QwenTextClient
            qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
            rerank_prompt = config.prompts.get("rerank", {})
            user_tpl = rerank_prompt.get("user", "")
            sys_prompt = rerank_prompt.get("system", "")

            if user_tpl:
                import re
                for r in all_results:
                    prompt = user_tpl.format(
                        query=query,
                        scene_caption=r.get("vlm_caption", ""),
                        transcript=r.get("transcript", ""),
                    )
                    resp = qwen.generate(prompt, system=sys_prompt)
                    nums = re.findall(r'\d+', resp or "")
                    if nums:
                        r["rerank_score"] = int(nums[0])
                    else:
                        r["rerank_score"] = r["score"] * 10

                all_results.sort(key=lambda x: -x.get("rerank_score", 0))
                logger.info("LLM 重排完成")
        except Exception as e:
            logger.warning(f"LLM 重排失败 (使用原始排序): {e}")

    return all_results, config


def _load_stage_prerequisites(video_id: str, stage: int):
    """为 test-stage 加载前置数据"""
    from vl.core.config import get_config
    from vl.core.models.scene import Scene
    from vl.core.helpers.json_utils import load_json

    config = get_config()
    output_dir = config.output_root

    def _check(path, label):
        if not os.path.isfile(path):
            typer.echo(f"错误: {label} 不存在: {path}", err=True)
            typer.echo(f"请先运行 Stage {stage - 1} 或更早的阶段。", err=True)
            raise typer.Exit(1)
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
            typer.echo(f"音频文件不存在，将从视频提取: {audio_path}")
        return {"scenes_json": scenes_json, "audio_path": audio_path}

    # Stage 3/4/5: 需要 scenes + transcripts + characters
    scenes_data = load_json(scenes_json)
    scenes = [Scene.from_dict(s) for s in scenes_data]

    # 尝试加载 metadata (enriched scenes)
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

    result = {
        "scenes": scenes,
        "scene_transcripts": scene_transcripts,
        "characters_info": characters_info,
    }

    return result


# ──────────────────────────────────────────────────────────
# CLI 命令
# ──────────────────────────────────────────────────────────

@app.command()
def index(
    video: str = typer.Argument(..., help="视频文件路径"),
    genre: str = typer.Option("movie", "--genre", "-g", help="视频类型: movie|tv|anime"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="是否从断点续跑"),
):
    """索引视频：场景分割 → 多维分析 → 构建索引"""
    typer.echo(_BANNER)

    if not os.path.isfile(video):
        typer.echo(f"错误: 视频文件不存在: {video}", err=True)
        raise typer.Exit(1)

    file_size_mb = os.path.getsize(video) / (1024 * 1024)
    logger.info(f"VideoLens v{VIDEOLENS_VERSION} 启动")
    logger.info(f"视频文件: {video} ({file_size_mb:.1f} MB)")

    from vl.pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(video, genre=genre)
    orchestrator.run(resume=resume)


@app.command()
def search(
    query: str = typer.Argument(..., help="搜索查询文本"),
    video: str = typer.Option("", "--video", "-v", help="视频ID (留空搜索所有)"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回结果数量"),
):
    """搜索场景：根据文本描述检索相关视频场景"""
    typer.echo(_BANNER)
    logger.info(f"搜索查询: \"{query}\" (top_k={top_k}, video={video or '全部'})")

    all_results, _ = _retrieve_scenes(query, video, top_k)

    if all_results is None:
        typer.echo("错误: 没有找到任何索引。请先运行 'videolens index' 建立索引。", err=True)
        raise typer.Exit(1)

    logger.info(f"搜索完成: 共找到 {len(all_results)} 个匹配结果")

    if not all_results:
        typer.echo("未找到匹配的场景。")
        return

    typer.echo(f"\n搜索结果 (共 {len(all_results)} 个):\n")
    typer.echo("-" * 70)
    for i, r in enumerate(all_results, 1):
        typer.echo(f"[{i}] 相似度: {r['score']:.3f} | 视频: {r.get('video_id', '?')}")
        start = r.get("start_time", 0)
        end = r.get("end_time", 0)
        typer.echo(f"    时间: {start:.1f}s - {end:.1f}s (时长: {end - start:.1f}s)")
        if r.get("vlm_caption"):
            caption = r["vlm_caption"][:200]
            typer.echo(f"    描述: {caption}...")
        if r.get("keyframe_paths"):
            typer.echo(f"    关键帧: {r['keyframe_paths'][0]}")
        typer.echo("-" * 70)


@app.command()
def qa(
    question: str = typer.Argument(..., help="问题"),
    video: str = typer.Option("", "--video", "-v", help="视频ID (留空搜索所有)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="检索场景数"),
):
    """视频问答：基于检索到的场景上下文回答问题"""
    typer.echo(_BANNER)
    logger.info(f"QA: \"{question}\" (top_k={top_k}, video={video or '全部'})")

    results, config = _retrieve_scenes(question, video, top_k)

    if results is None or not results:
        typer.echo("错误: 没有找到相关场景。请先运行 'videolens index' 建立索引。", err=True)
        raise typer.Exit(1)

    # 拼接上下文
    context_parts = []
    for i, r in enumerate(results, 1):
        start = r.get("start_time", 0)
        end = r.get("end_time", 0)
        part = f"[场景 {i}] {start:.1f}s - {end:.1f}s"
        if r.get("structured_caption"):
            part += f"\n  视觉: {json.dumps(r['structured_caption'], ensure_ascii=False)}"
        if r.get("vlm_caption"):
            part += f"\n  描述: {r['vlm_caption'][:300]}"
        if r.get("transcript"):
            part += f"\n  台词: {r['transcript'][:300]}"
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    # 调用 LLM
    if not config.dashscope_api_key:
        typer.echo("错误: 未配置 DASHSCOPE_API_KEY，无法进行问答。", err=True)
        raise typer.Exit(1)

    from vl.core.llm.qwen_text import QwenTextClient
    qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

    prompts = config.prompts.get("qa_answer", {})
    user_tpl = prompts.get("user", "")
    sys_prompt = prompts.get("system", "")

    if not user_tpl:
        typer.echo("错误: qa_answer prompt 未配置", err=True)
        raise typer.Exit(1)

    prompt = user_tpl.format(question=question, context=context)

    logger.info("正在生成回答...")
    answer = qwen.generate(prompt, system=sys_prompt)

    if answer:
        typer.echo(f"\n{'=' * 70}")
        typer.echo(f"问题: {question}")
        typer.echo(f"{'=' * 70}")
        typer.echo(f"\n{answer}")
        typer.echo(f"\n{'=' * 70}")
        typer.echo(f"参考场景: {len(results)} 个")
        for i, r in enumerate(results, 1):
            start = r.get("start_time", 0)
            end = r.get("end_time", 0)
            typer.echo(f"  [{i}] {r.get('video_id', '?')} {start:.1f}s-{end:.1f}s (相似度: {r['score']:.3f})")
        typer.echo(f"{'=' * 70}")
    else:
        typer.echo("未能生成回答。", err=True)


@app.command()
def analyze(
    video: str = typer.Argument(..., help="视频ID"),
    analysis_type: str = typer.Option("summary", "--type", "-t",
                                       help="分析类型: summary|characters|timeline"),
):
    """视频分析：生成摘要、角色分析或时间线"""
    typer.echo(_BANNER)
    logger.info(f"分析视频: {video} (类型: {analysis_type})")

    from vl.core.config import get_config
    from vl.core.helpers.json_utils import load_json

    config = get_config()

    # 加载 doc_store
    doc_path = os.path.join(config.output_root, "stage3_captions", video, "doc_store.json")
    if not os.path.isfile(doc_path):
        typer.echo(f"错误: 未找到视频 {video} 的索引数据。请先运行 'videolens index'。", err=True)
        raise typer.Exit(1)

    docs = load_json(doc_path)

    # 拼接上下文
    context_parts = []
    for doc in docs:
        start = doc.get("start_time", 0)
        end = doc.get("end_time", 0)
        part = f"[{start:.1f}s - {end:.1f}s]"
        if doc.get("structured_caption"):
            part += f" {json.dumps(doc['structured_caption'], ensure_ascii=False)}"
        if doc.get("transcript"):
            part += f" 台词: {doc['transcript'][:200]}"
        context_parts.append(part)

    context = "\n".join(context_parts)

    # 选择 prompt
    prompt_key = f"analyze_{analysis_type}"
    prompts = config.prompts.get(prompt_key, {})
    user_tpl = prompts.get("user", "")
    sys_prompt = prompts.get("system", "")

    if not config.dashscope_api_key:
        typer.echo("错误: 未配置 DASHSCOPE_API_KEY，无法进行分析。", err=True)
        raise typer.Exit(1)

    if not user_tpl:
        typer.echo(f"错误: {prompt_key} prompt 未配置", err=True)
        raise typer.Exit(1)

    from vl.core.llm.qwen_text import QwenTextClient
    qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)

    prompt = user_tpl.format(
        video_id=video,
        scene_count=len(docs),
        context=context,
    )

    logger.info(f"正在生成{analysis_type}分析...")
    result = qwen.generate(prompt, system=sys_prompt)

    if result:
        typer.echo(f"\n{'=' * 70}")
        typer.echo(f"视频分析: {video} ({analysis_type})")
        typer.echo(f"{'=' * 70}")
        typer.echo(f"\n{result}")
        typer.echo(f"\n{'=' * 70}")
    else:
        typer.echo("未能生成分析结果。", err=True)


@app.command("test-stage")
def test_stage(
    video_id: str = typer.Argument(..., help="视频ID (如 052)"),
    stage: int = typer.Option(1, "--stage", "-s", help="要测试的阶段 (1-6)"),
):
    """单独测试流水线中的某个阶段"""
    typer.echo(_BANNER)

    if stage not in (1, 2, 3, 4, 5):
        typer.echo(f"错误: 无效的阶段 {stage}，请选择 1-5。", err=True)
        raise typer.Exit(1)

    stage_names = {
        1: "场景分割",
        2: "多模态场景理解",
        3: "视觉语义理解 + 索引",
        4: "事件提取",
        5: "结构化知识库生成",
    }
    typer.echo(f"测试 Stage {stage}: {stage_names[stage]} (视频: {video_id})")
    typer.echo("-" * 50)

    config = None

    try:
        prereq = _load_stage_prerequisites(video_id, stage)
        config = prereq.get("config") or _get_config()
        output_dir = config.output_root
        video_title = video_id
        genre = "movie"

        import time
        t0 = time.time()

        if stage == 1:
            from vl.pipeline.stage1_ingestion import run_stage1
            scenes = run_stage1(
                video_path=prereq["video_path"],
                output_dir=os.path.join(output_dir, "stage1_scenes", video_id),
                config=config,
            )
            # 保存 scenes.json
            from vl.core.helpers.json_utils import save_json
            save_json(
                [s.to_dict() for s in scenes],
                os.path.join(output_dir, "stage1_scenes", video_id, "scenes.json"),
            )
            typer.echo(f"\n结果: 检测到 {len(scenes)} 个场景")

        elif stage == 2:
            from vl.pipeline.stage2_analysis import run_stage2
            from vl.core.models.scene import Scene
            from vl.core.helpers.json_utils import load_json
            from vl.core.helpers.ffmpeg import extract_audio

            scenes_data = load_json(prereq["scenes_json"])
            scenes = [Scene.from_dict(s) for s in scenes_data]

            # 确保音频存在
            audio_path = prereq["audio_path"]
            if not os.path.isfile(audio_path):
                typer.echo("音频不存在，从视频提取...")
                video_path = os.path.join(config.data_root, "videos", f"{video_id}.mp4")
                if not os.path.isfile(video_path):
                    typer.echo(f"错误: 视频文件不存在: {video_path}", err=True)
                    raise typer.Exit(1)
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                extract_audio(video_path, audio_path)

            scene_transcripts = run_stage2(
                video_path=os.path.join(config.data_root, "videos", f"{video_id}.mp4"),
                video_id=video_id,
                scenes=scenes,
                audio_path=audio_path,
                output_dir=output_dir,
                config=config,
            )
            total_segs = sum(len(v) for v in scene_transcripts.values())
            typer.echo(f"\n结果: {len(scene_transcripts)} 个场景有台词, 共 {total_segs} 个转录片段")

        elif stage == 3:
            from vl.pipeline.stage3_understanding import run_stage3
            run_stage3(
                video_id=video_id,
                scenes=prereq["scenes"],
                scene_transcripts=prereq["scene_transcripts"],
                output_dir=output_dir,
                config=config,
            )
            captioned = sum(1 for s in prereq["scenes"] if s.structured_caption)
            typer.echo(f"\n结果: {len(prereq['scenes'])} 个场景, {captioned} 个有结构化描述")

        elif stage == 4:
            from vl.pipeline.stage4_event_builder import run_stage4
            events = run_stage4(
                video_id=video_id,
                scenes=prereq["scenes"],
                scene_transcripts=prereq["scene_transcripts"],
                output_dir=output_dir,
                config=config,
            )
            typer.echo(f"\n结果: 提取 {len(events)} 个事件")

        elif stage == 5:
            from vl.pipeline.stage5_knowledge import run_stage5
            kb = run_stage5(
                video_id=video_id,
                video_title=video_title,
                scenes=prereq["scenes"],
                scene_transcripts=prereq["scene_transcripts"],
                output_dir=output_dir,
                config=config,
            )
            events_count = len(kb.get(video_title, {}))
            sub_count = sum(len(v) for v in kb.get(video_title, {}).values())
            typer.echo(f"\n结果: {events_count} 个大事件, {sub_count} 个小事件")

        elapsed = time.time() - t0
        typer.echo(f"\n耗时: {elapsed:.1f}s")
        typer.echo("-" * 50)
        typer.echo(f"Stage {stage} 测试完成 ✓")

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Stage {stage} 测试失败: {e}", exc_info=True)
        typer.echo(f"\nStage {stage} 测试失败: {e}", err=True)
        raise typer.Exit(1)


def _get_config():
    from vl.core.config import get_config
    return get_config()


if __name__ == "__main__":
    app()
