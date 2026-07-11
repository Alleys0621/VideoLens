"use client";

import { useEffect, useRef } from "react";
import Artplayer from "artplayer";
import { useKeyframeSeek, type KeyframeMeta } from "@/hooks/useKeyframeSeek";
import { Sparkles, Play } from "lucide-react";
import { cn } from "@/lib/utils";

export function VideoPlayer({
  videoDir,
  onVideoDirChange,
}: {
  videoDir: string;
  onVideoDirChange: (dir: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const artRef = useRef<Artplayer | null>(null);
  const { keyframes, reasoning } = useKeyframeSeek();

  const videoSrc = videoDir ? `/api/video/${videoDir}` : "";

  // === 推理数据解析 (保留原有) ===
  const intentMeta: Record<string, { label: string; color: string }> = {
    kb: { label: "知识库检索", color: "bg-blue-500/15 text-blue-400" },
    kb_meta: { label: "剧情概要", color: "bg-purple-500/15 text-purple-400" },
    chitchat: { label: "闲聊", color: "bg-amber-500/15 text-amber-400" },
    refuse: { label: "拒答", color: "bg-rose-500/15 text-rose-400" },
  };
  const rIntent = reasoning?.intent ?? "";
  const rMeta = intentMeta[rIntent] ?? {
    label: rIntent,
    color: "bg-zinc-700 text-zinc-400",
  };
  const rRetrieved = ((reasoning as Record<string, unknown>)?.retrieved ??
    []) as { title?: string; event_id?: string; score?: number }[];
  const rScore = (reasoning as Record<string, unknown>)?.top_score;
  const rTimings = (reasoning as Record<string, unknown>)?.timings as
    | { retrieval_ms?: number; llm_ms?: number; total_ms?: number }
    | undefined;

  // === Artplayer 初始化 (切换视频时销毁重建) ===
  // 内置: 鼠标移入控制条渐显/移出淡出、暂停时中间大播放按钮、进度条 hover 预览
  useEffect(() => {
    if (!containerRef.current || !videoSrc) return;

    const art = new Artplayer({
      container: containerRef.current,
      url: videoSrc,
      autoplay: true, // 选集后自动播放 (浏览器策略允许时; 被拦截则点击即播)
      volume: 0.7,
      autoSize: false,
      autoMini: false,
      screenshot: false,
      pip: true, // 画中画
      setting: true, // 设置面板 (倍速等)
      playbackRate: true, // 倍速
      fullscreen: true,
      fullscreenWeb: true, // 网页全屏
      hotkey: true, // 快捷键 (空格暂停/方向键 seek)
      mutex: true, // 同时只播一个
      backdrop: true,
      playsInline: true,
      airplay: true,
      theme: "#a78bfa", // 淡紫色 (violet-400) — 进度条/高亮/按钮主题色
      lang: "zh-cn",
      moreVideoAttr: {
        crossOrigin: "anonymous",
      },
    });
    artRef.current = art;

    return () => {
      art.destroy(false); // false = 不删除容器 DOM
      artRef.current = null;
    };
  }, [videoSrc]);

  return (
    <div className="flex h-full w-full flex-col bg-zinc-950">
      {/* === 视频区 (Artplayer 挂载点, 占满方框卡片) === */}
      <div className="relative min-h-0 flex-1 overflow-hidden">
        {videoSrc ? (
          <div ref={containerRef} className="art-container h-full w-full" />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <div className="flex flex-col items-center justify-center gap-3 text-zinc-500">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-zinc-800/80">
                <Play className="h-7 w-7 text-zinc-600" />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-zinc-400">
                  选择一集开始陪看
                </p>
                <p className="mt-1 text-xs text-zinc-600">
                  Alleys会在你看视频时聊剧情
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* === 推理信息条 (保留原有) === */}
      {rIntent && (
        <div className="flex items-center gap-3 border-t border-zinc-800 bg-zinc-900/90 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-indigo-500 to-violet-500">
              <Sparkles className="h-3 w-3 text-white" />
            </div>
            <span className="text-xs font-medium text-zinc-300">Alleys的推理</span>
          </div>

          <span
            className={cn(
              "rounded-full px-2 py-0.5 text-[10px] font-medium",
              rMeta.color,
            )}
          >
            {rMeta.label}
          </span>

          <span className="text-[10px] text-zinc-500">
            {rIntent === "kb" && typeof rScore === "number" && (
              <span className="mr-2">相关度 {rScore}</span>
            )}
            {rTimings && (
              <>
                {rTimings.retrieval_ms !== undefined && (
                  <span>检索 {rTimings.retrieval_ms}ms</span>
                )}
                {rTimings.llm_ms !== undefined && (
                  <span className="ml-1">· LLM {rTimings.llm_ms}ms</span>
                )}
              </>
            )}
          </span>

          <div className="flex-1" />

          {keyframes.length > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="mr-1 text-[10px] text-zinc-500">定位</span>
              {keyframes.map((kf: KeyframeMeta, i: number) => (
                <button
                  key={i}
                  onClick={() => {
                    if (artRef.current)
                      artRef.current.currentTime = kf.timestamp;
                  }}
                  className="rounded-md border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300 transition-colors hover:border-indigo-500/50 hover:text-indigo-300"
                >
                  {kf.timestamp.toFixed(1)}s
                </button>
              ))}
            </div>
          )}

          {rRetrieved.length > 0 && keyframes.length === 0 && (
            <div className="flex items-center gap-1.5 text-[10px] text-zinc-500">
              <span>来源:</span>
              {rRetrieved.slice(0, 2).map((e, i) => (
                <span
                  key={i}
                  className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400"
                >
                  {e.title || e.event_id}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
