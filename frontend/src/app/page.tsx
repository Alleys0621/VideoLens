"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { VideoPlayer } from "@/components/video-player/VideoPlayer";
import { TopNav, LeftSidebar } from "@/components/layout";
import {
  Sheet,
  SheetContent,
  SheetTitle,
} from "@/components/ui/sheet";
import React from "react";
import { useQueryState } from "nuqs";
import { useMediaQuery } from "@/hooks/useMediaQuery";

export default function DemoPage(): React.ReactNode {
  // videoDir 通过 URL query state 在 TopNav / LeftSidebar / VideoPlayer / Thread 之间共享
  const [videoDir, setVideoDir] = useQueryState("videoDir", {
    defaultValue: "",
  });

  // 响应式断点: ≥1024px 走桌面三栏, 否则走「中+右两栏 + 左栏抽屉」(iPad 竖屏)
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const [leftOpen, setLeftOpen] = React.useState(false);
  // 视频播放时间戳 (VideoPlayer 写, Thread 读并发给后端做邻域对白检索)
  const videoTimeRef = React.useRef(0);
  // 视频外部控制句柄 (ASR 录音时通过它暂停/恢复视频, 防回声)
  const videoControlRef = React.useRef<{
    pause: () => void;
    resume: () => void;
    isPaused: () => boolean;
    duckVolume: () => void;
    restoreVolume: () => void;
  } | null>(null);

  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <ThreadProvider>
        <StreamProvider>
          {/* 三栏布局: TopNav + [LeftSidebar | VideoPlayer | Thread] */}
          {/* 灰色画布 + 三张白色圆角浮层卡片 (NotebookLM 风格) */}
          <div className="flex h-screen w-full flex-col bg-[#F0F2F5]">
            <TopNav
              videoDir={videoDir}
              onToggleSidebar={
                isDesktop ? undefined : () => setLeftOpen((p) => !p)
              }
            />

            {/* px-4 提供每边 16px 留白; lg: 起进入 iPad 横屏; min-[1700px]: 外接显示器 */}
            <div className="flex min-h-0 flex-1 px-4 py-3">
              <div className="flex h-full w-full min-h-0 gap-4">
                {/* === 桌面: 左栏常驻 (Tailwind 断点) ===
                    lg (≥1024): iPad 横屏 220px
                    xl (≥1280): 笔记本 260px
                    min-[1700px]: 外接 320px */}
                {isDesktop && (
                  <div className="flex h-full w-[220px] flex-shrink-0 overflow-hidden rounded-xl bg-white shadow-soft-md ring-1 ring-zinc-200/60 xl:w-[260px] min-[1700px]:w-[320px]">
                    <LeftSidebar
                      videoDir={videoDir}
                      onVideoDirChange={setVideoDir}
                    />
                  </div>
                )}

                {/* === iPad 竖屏: 左栏抽屉 (Sheet) === */}
                {!isDesktop && (
                  <Sheet open={leftOpen} onOpenChange={setLeftOpen}>
                    <SheetContent
                      side="left"
                      className="w-[300px] max-w-[85vw] p-0"
                    >
                      {/* radix-dialog 要求必须有 title, sr-only 视觉隐藏用于 a11y */}
                      <div className="sr-only">
                        <SheetTitle>剧集列表</SheetTitle>
                      </div>
                      <LeftSidebar
                        videoDir={videoDir}
                        onVideoDirChange={(dir) => {
                          setVideoDir(dir);
                          setLeftOpen(false);
                        }}
                      />
                    </SheetContent>
                  </Sheet>
                )}

                {/* 中卡: 方框 (国内视频站风格, 无圆角), flex-1 自然填满 */}
                <div className="flex h-full min-w-0 flex-1 overflow-hidden shadow-soft-md ring-1 ring-black/10">
                  <VideoPlayer
                    videoDir={videoDir}
                    onVideoDirChange={setVideoDir}
                    videoTimeRef={videoTimeRef}
                    videoControlRef={videoControlRef}
                  />
                </div>

                {/* 右卡: 笔记本 380px / 外接 460px / iPad 横屏 340px */}
                <div className="flex h-full w-[340px] flex-shrink-0 overflow-hidden rounded-xl bg-white shadow-soft-md ring-1 ring-zinc-200/60 xl:w-[380px] min-[1700px]:w-[460px]">
                  <Thread
                    videoTimeRef={videoTimeRef}
                    videoControlRef={videoControlRef}
                  />
                </div>
              </div>
            </div>
          </div>
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}
