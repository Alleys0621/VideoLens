"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Artplayer from "artplayer";
import { useKeyframeSeek, type KeyframeMeta } from "@/hooks/useKeyframeSeek";
import { Sparkles, Play } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

/* -------------------------------------------------------------------------- */
/* 播放进度持久化 (LangGraph Store, namespace=["playback", user_id])          */
/* -------------------------------------------------------------------------- */

/** GET 当前用户该视频的播放进度. 无记录或失败返回 null. */
async function fetchPlayback(videoDir: string): Promise<{ position: number; duration: number | null } | null> {
  try {
    const res = await fetch(
      `/api/playback?video_dir=${encodeURIComponent(videoDir)}`,
    );
    if (!res.ok) return null;
    const data = await res.json();
    if (typeof data.position !== "number" || data.position <= 5) return null;
    return { position: data.position, duration: data.duration ?? null };
  } catch {
    return null;
  }
}

/** 节流保存: 距离上次保存至少 SAVE_THROTTLE_MS 毫秒 */
const SAVE_THROTTLE_MS = 10_000;
let lastSaveAt = 0;

/** POST 保存进度. 失败静默 (不影响播放体验). */
function savePlayback(
  videoDir: string,
  position: number,
  duration: number | null,
  opts: { force?: boolean; completed?: boolean } = {},
) {
  if (!videoDir || !Number.isFinite(position) || position < 0) return;
  const now = Date.now();
  if (!opts.force && now - lastSaveAt < SAVE_THROTTLE_MS) return;
  lastSaveAt = now;
  const body = JSON.stringify({
    video_dir: videoDir,
    position,
    duration: Number.isFinite(duration as number) ? duration : null,
    completed: opts.completed,
  });
  // fire-and-forget; pagehide 场景用 sendBeacon 保证发出
  if (typeof navigator !== "undefined" && navigator.sendBeacon) {
    const blob = new Blob([body], { type: "application/json" });
    if (navigator.sendBeacon("/api/playback", blob)) return;
  }
  fetch("/api/playback", { method: "POST", body, headers: { "Content-Type": "application/json" } }).catch(() => {});
}

/** 格式化秒为 mm:ss, 用于 "从 XX:XX 继续" 提示 */
function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// === 观看状态上报 (实时坐标, agent retrieve_kb 用) ===
// 与 savePlayback 区别: 这里存"此刻", 高频 (5s 节流); 那里存"上次到哪了", 10s 节流.
const WATCH_THROTTLE_MS = 5_000;
let lastWatchAt = 0;

function reportWatching(
  videoDir: string,
  videoTime: number,
  isPlaying: boolean,
  opts: { force?: boolean } = {},
) {
  if (!videoDir || !Number.isFinite(videoTime) || videoTime < 0) return;
  const now = Date.now();
  if (!opts.force && now - lastWatchAt < WATCH_THROTTLE_MS) return;
  lastWatchAt = now;
  const body = JSON.stringify({ video_dir: videoDir, video_time: videoTime, is_playing: isPlaying });
  if (typeof navigator !== "undefined" && navigator.sendBeacon) {
    const blob = new Blob([body], { type: "application/json" });
    if (navigator.sendBeacon("/api/watching", blob)) return;
  }
  fetch("/api/watching", { method: "POST", body, headers: { "Content-Type": "application/json" } }).catch(() => {});
}

export function VideoPlayer({
  videoDir,
  onVideoDirChange,
  videoTimeRef,
  videoControlRef,
}: {
  videoDir: string;
  onVideoDirChange: (dir: string) => void;
  videoTimeRef?: { current: number };
  /**
   * 外部控制句柄: pause/resume 用于 ASR 录音防回声,
   * duckVolume/restoreVolume 用于 TTS 播放时降低视频音量 (2s 淡入淡出).
   */
  videoControlRef?: {
    current: {
      pause: () => void;
      resume: () => void;
      isPaused: () => boolean;
      duckVolume: () => void;
      restoreVolume: () => void;
    } | null;
  };
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const artRef = useRef<Artplayer | null>(null);
  const volumeRampRef = useRef<number | null>(null); // 音量渐变动画 frame id
  const { keyframes, reasoning } = useKeyframeSeek();

  const actualSrc = useMemo(() => {
    if (!videoDir) return "";
    const videoBase =
      process.env.NEXT_PUBLIC_VIDEO_SERVER_URL || "http://127.0.0.1:9802";
    return `${videoBase}/${videoDir}`;
  }, [videoDir]);

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
    if (!containerRef.current || !actualSrc) return;

    const art = new Artplayer({
      container: containerRef.current,
      url: actualSrc,
      autoplay: true,
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
      fastForward: false, // 关闭长按倍速 (iPad 默认会触发, 干扰正常观看)
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

    // 实时更新播放时间戳到 ref (Thread 发送消息时读取, 传后端做邻域对白检索)
    art.on("video:timeupdate", () => {
      const t = art.currentTime || 0;
      if (videoTimeRef) videoTimeRef.current = t;
      savePlayback(videoDir, t, art.duration);
      reportWatching(videoDir, t, true);
    });

    // 暂停时立即保存 (force=true 跳过节流)
    art.on("video:pause", () => {
      savePlayback(videoDir, art.currentTime || 0, art.duration, { force: true });
      reportWatching(videoDir, art.currentTime || 0, false, { force: true });
    });

    // seek 完成: 立即上报新位置 (让 agent 知道用户跳到了哪)
    art.on("video:seeked", () => {
      reportWatching(videoDir, art.currentTime || 0, !art.video.paused, { force: true });
    });

    // 播放结束: 标记完成, 下次从头
    art.on("video:ended", () => {
      savePlayback(videoDir, 0, art.duration, { force: true, completed: true });
      reportWatching(videoDir, 0, false, { force: true });
    });

    // 加载已有进度并 seek (用户上次离开的位置)
    let seeked = false;
    art.on("ready", async () => {
      if (seeked) return;
      seeked = true;
      const playback = await fetchPlayback(videoDir);
      if (playback && playback.position > 5) {
        try {
          art.currentTime = playback.position;
          toast.info(`从 ${fmtTime(playback.position)} 继续`);
        } catch {
          /* noop */
        }
      }
    });

    // 暴露 pause / resume / isPaused / duckVolume / restoreVolume 给外部
    if (videoControlRef) {
      const rampVolume = (target: number, duration = 2000) => {
        const video = art.video as HTMLVideoElement;
        const start = video.volume;
        if (Math.abs(start - target) < 0.01) return;
        if (volumeRampRef.current) cancelAnimationFrame(volumeRampRef.current);
        const t0 = performance.now();
        const step = (now: number) => {
          const p = Math.min((now - t0) / duration, 1);
          // clamp 到 [0, 1] 防止浮点误差导致 volume > 1 报错
          video.volume = Math.max(0, Math.min(1, start + (target - start) * p));
          if (p < 1) {
            volumeRampRef.current = requestAnimationFrame(step);
          } else {
            volumeRampRef.current = null;
          }
        };
        volumeRampRef.current = requestAnimationFrame(step);
      };

      videoControlRef.current = {
        pause: () => {
          try { art.pause(); } catch { /* noop */ }
        },
        resume: () => {
          try {
            if (!art.video.paused) return;
            art.play();
          } catch { /* noop */ }
        },
        isPaused: () => art.video.paused,
        duckVolume: () => rampVolume(0.15),   // TTS 播放时降到 15%
        restoreVolume: () => rampVolume(0.7), // TTS 停止后恢复到默认 70%
      };
    }

    // 页面关闭时用 sendBeacon 强制保存 (避免用户关浏览器丢失最后进度)
    const onPageHide = () => {
      const a = artRef.current;
      if (a && !a.video.ended) {
        savePlayback(videoDir, a.currentTime || 0, a.duration, { force: true });
      }
    };
    window.addEventListener("pagehide", onPageHide);

    return () => {
      // unmount 时保存最后位置
      const a = artRef.current;
      if (a && !a.video.ended) {
        savePlayback(videoDir, a.currentTime || 0, a.duration, { force: true });
      }
      window.removeEventListener("pagehide", onPageHide);
      art.destroy(false);
      artRef.current = null;
      if (videoControlRef) videoControlRef.current = null;
    };
  }, [actualSrc, videoTimeRef, videoControlRef]);

  return (
    <div className="flex h-full w-full flex-col bg-zinc-950">
      {/* === 视频区 (Artplayer 挂载点, 占满方框卡片) === */}
      <div className="relative min-h-0 flex-1 overflow-hidden">
        {actualSrc ? (
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
                  className="flex min-h-[32px] items-center rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-[11px] text-zinc-300 transition-colors hover:border-indigo-500/50 hover:text-indigo-300"
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
