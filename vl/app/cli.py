"""VideoLens CLI - 命令行接口"""

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

    # 获取视频文件大小
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
    import json
    from vl.core.config import get_config
    from vl.vision.clip_encoder import CLIPEncoder
    from vl.store.vector_store import VectorStore

    typer.echo(_BANNER)
    logger.info(f"搜索查询: \"{query}\" (top_k={top_k}, video={video or '全部'})")

    config = get_config()
    clip = CLIPEncoder(model_name=config.model_clip)

    # --- Phase 2: 查询扩展 ---
    queries = [query]
    if config.dashscope_api_key:
        try:
            from vl.core.llm.qwen_text import QwenTextClient
            qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
            expand_prompt = config.prompts.get("query_expand", {})
            user_tpl = expand_prompt.get("user", "")
            sys_prompt = expand_prompt.get("system", "")
            if user_tpl:
                expanded = qwen.generate(user_tpl.format(query=query), system_prompt=sys_prompt)
                if expanded:
                    extra = [w.strip() for w in expanded.split() if w.strip()]
                    queries.extend(extra)
                    logger.info(f"查询扩展: {query} -> {queries}")
        except Exception as e:
            logger.warning(f"查询扩展失败 (使用原始查询): {e}")

    # 编码查询
    query_vecs = clip.encode_texts(queries)
    # 平均所有查询向量
    query_vec = np.mean(query_vecs, axis=0)
    query_vec = query_vec / np.linalg.norm(query_vec)

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

    # --- Phase 2: LLM 重排 ---
    if config.retrieval_rerank and config.dashscope_api_key and all_results:
        try:
            from vl.core.llm.qwen_text import QwenTextClient
            qwen = QwenTextClient(model=config.model_text, api_key=config.dashscope_api_key)
            rerank_prompt = config.prompts.get("rerank", {})
            user_tpl = rerank_prompt.get("user", "")
            sys_prompt = rerank_prompt.get("system", "")

            if user_tpl:
                for r in all_results:
                    prompt = user_tpl.format(
                        query=query,
                        scene_caption=r.get("vlm_caption", ""),
                        transcript=r.get("transcript", ""),
                    )
                    resp = qwen.generate(prompt, system_prompt=sys_prompt)
                    # 提取数字分数
                    import re
                    nums = re.findall(r'\d+', resp or "")
                    if nums:
                        r["rerank_score"] = int(nums[0])
                    else:
                        r["rerank_score"] = r["score"] * 10

                all_results.sort(key=lambda x: -x.get("rerank_score", 0))
                logger.info("LLM 重排完成")
        except Exception as e:
            logger.warning(f"LLM 重排失败 (使用原始排序): {e}")

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


if __name__ == "__main__":
    app()
