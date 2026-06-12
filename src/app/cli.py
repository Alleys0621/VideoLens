"""VideoLens CLI - 命令行接口"""

import os

import typer

from src.core.logging import get_logger

logger = get_logger()

VIDEOLENS_VERSION = "1.0.0"

_BANNER = f"""
  __   __     _       ___      _
  \\ \\ / /__ _| |_ ___| __)_ __| |
   \\ V / _` |  _/ _ \\ __| '__| |
    | | (_| | ||  __/ |_| |  | |
    |_|\\__,_|\\__\\___/\\__|_|  |_|   v{VIDEOLENS_VERSION}
"""

app = typer.Typer(name="videolens", help="VideoLens - 影视视频内容分析系统")


@app.command()
def run(
    video: str = typer.Argument(..., help="视频目录名, 如 '052 鸟蛋之争'"),
    stage: int = typer.Option(0, "--stage", "-s", help="运行到哪个 stage (0=全部, 1/2/3)"),
    skip_theme: bool = typer.Option(False, "--skip-theme", help="跳过片头/片尾曲检测"),
    chunk_dur: int = typer.Option(60, "--chunk", "-c", help="Omni chunk 时长 (秒)"),
    vp_threshold: float = typer.Option(0.0, "--vp-threshold", help="声纹置信度阈值, 低于此值标为 '路人' (0=不过滤)"),
):
    """运行 Pipeline"""
    typer.echo(_BANNER)

    from src.pipeline.orchestrator import resolve_video_path
    video_path = resolve_video_path(video)
    if not os.path.isfile(video_path):
        typer.echo(f"错误: 视频文件不存在: {video_path}", err=True)
        raise typer.Exit(1)

    from src.pipeline.orchestrator import run_pipeline
    run_pipeline(video, stage=stage, skip_theme=skip_theme, chunk_dur=chunk_dur, vp_threshold=vp_threshold)


@app.command()
def test(
    video: str = typer.Argument(..., help="视频目录名"),
    stage: int = typer.Option(1, "--stage", "-s", help="要测试的阶段 (1/2/3)"),
):
    """测试单个 Stage"""
    typer.echo(_BANNER)

    if stage not in (1, 2, 3):
        typer.echo(f"错误: 无效的阶段 {stage}，请选择 1/2/3。", err=True)
        raise typer.Exit(1)

    stage_names = {1: "音频处理", 2: "视觉处理", 3: "结构化知识库"}
    typer.echo(f"测试 Stage {stage}: {stage_names[stage]} (视频: {video})")

    from src.pipeline.orchestrator import run_pipeline
    run_pipeline(video, stage=stage)


if __name__ == "__main__":
    app()
