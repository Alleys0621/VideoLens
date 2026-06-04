"""VideoLens CLI - 命令行接口 (薄层，委托到 services)"""

import os

import typer

from vl.core.logging import get_logger
from vl.core.cost import get_cost_tracker, reset_cost_tracker

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


def _init_tracker():
    """初始化代价追踪器 (加载配置中的定价)"""
    from vl.core.config import get_config
    config = get_config()
    reset_cost_tracker()
    tracker = get_cost_tracker()
    tracker.configure_pricing(config.pricing)
    return tracker


def _print_cost_report(tracker):
    """输出代价报告"""
    report = tracker.report()
    if tracker.total_calls > 0:
        typer.echo(report)


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

    # 流水线跑完输出代价报告
    tracker = get_cost_tracker()
    _print_cost_report(tracker)


@app.command()
def search(
    query: str = typer.Argument(..., help="搜索查询文本"),
    video: str = typer.Option("", "--video", "-v", help="视频ID (留空搜索所有)"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回结果数量"),
):
    """搜索场景：根据文本描述检索相关视频场景"""
    typer.echo(_BANNER)
    logger.info(f"搜索查询: \"{query}\" (top_k={top_k}, video={video or '全部'})")

    tracker = _init_tracker()

    from vl.services.search import search_scenes

    all_results, _ = search_scenes(query, video, top_k)

    if all_results is None:
        typer.echo("错误: 没有找到任何索引。请先运行 'videolens index' 建立索引。", err=True)
        raise typer.Exit(1)

    logger.info(f"搜索完成: 共找到 {len(all_results)} 个匹配结果")

    if not all_results:
        typer.echo("未找到匹配的场景。")
        _print_cost_report(tracker)
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

    _print_cost_report(tracker)


@app.command()
def qa(
    question: str = typer.Argument(..., help="问题"),
    video: str = typer.Option("", "--video", "-v", help="视频ID (留空搜索所有)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="检索场景数"),
):
    """视频问答：基于检索到的场景上下文回答问题"""
    typer.echo(_BANNER)
    logger.info(f"QA: \"{question}\" (top_k={top_k}, video={video or '全部'})")

    tracker = _init_tracker()

    from vl.services.qa import answer_question

    try:
        answer, results = answer_question(question, video, top_k)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

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

    _print_cost_report(tracker)


@app.command()
def analyze(
    video: str = typer.Argument(..., help="视频ID"),
    analysis_type: str = typer.Option("summary", "--type", "-t",
                                       help="分析类型: summary|characters|timeline"),
):
    """视频分析：生成摘要、角色分析或时间线"""
    typer.echo(_BANNER)
    logger.info(f"分析视频: {video} (类型: {analysis_type})")

    tracker = _init_tracker()

    from vl.services.analysis import analyze_video

    try:
        result = analyze_video(video, analysis_type)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"错误: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{'=' * 70}")
    typer.echo(f"视频分析: {video} ({analysis_type})")
    typer.echo(f"{'=' * 70}")
    typer.echo(f"\n{result}")
    typer.echo(f"\n{'=' * 70}")

    _print_cost_report(tracker)


@app.command("test-stage")
def test_stage(
    video_id: str = typer.Argument(..., help="视频ID (如 052)"),
    stage: int = typer.Option(1, "--stage", "-s", help="要测试的阶段 (1-5)"),
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

    tracker = _init_tracker()

    from vl.services.test_stage import run_test_stage

    try:
        result = run_test_stage(video_id, stage)
        typer.echo(f"\n结果: {result}")
        typer.echo("-" * 50)
        typer.echo(f"Stage {stage} 测试完成 ✓")
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"\n错误: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Stage {stage} 测试失败: {e}", exc_info=True)
        typer.echo(f"\nStage {stage} 测试失败: {e}", err=True)
        raise typer.Exit(1)

    _print_cost_report(tracker)


if __name__ == "__main__":
    app()
