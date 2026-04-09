"""VideoLens CLI - 命令行接口"""

import sys

import typer

app = typer.Typer(name="videolens", help="VideoLens - 影视视频内容分析与检索")


@app.command()
def index(
    video: str = typer.Argument(..., help="视频文件路径"),
    genre: str = typer.Option("movie", "--genre", "-g", help="视频类型: movie|tv|anime"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="是否从断点续跑"),
):
    """索引视频：场景分割 → 多维分析 → 构建索引"""
    import os
    if not os.path.isfile(video):
        typer.echo(f"错误: 视频文件不存在: {video}", err=True)
        raise typer.Exit(1)

    from vl_pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(video, genre=genre)
    orchestrator.run(resume=resume)


@app.command()
def search(
    query: str = typer.Argument(..., help="搜索查询文本"),
    video: str = typer.Option("", "--video", "-v", help="视频ID (留空搜索所有)"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回结果数量"),
):
    """搜索场景：根据文本描述检索相关视频场景"""
    import os
    import json
    from vl_core.config import get_config
    from vl_vision.clip_encoder import CLIPEncoder
    from vl_store.vector_store import VectorStore

    config = get_config()
    clip = CLIPEncoder(model_name=config.model_clip)

    # 编码查询
    query_vec = clip.encode_text(query)

    # 查找可用的索引
    index_root = os.path.join(config.output_root, "index")
    if not os.path.isdir(index_root):
        typer.echo("错误: 没有找到任何索引。请先运行 'videolens index' 建立索引。", err=True)
        raise typer.Exit(1)

    # 搜索
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

    # 排序并输出
    all_results.sort(key=lambda x: -x["score"])
    all_results = all_results[:top_k]

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


if __name__ == "__main__":
    app()
