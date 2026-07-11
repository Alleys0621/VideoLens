"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { VideoPlayer } from "@/components/video-player/VideoPlayer";
import { TopNav, LeftSidebar } from "@/components/layout";
import { Toaster } from "@/components/ui/sonner";
import React from "react";
import { useQueryState } from "nuqs";

export default function DemoPage(): React.ReactNode {
  // videoDir 通过 URL query state 在 TopNav / LeftSidebar / VideoPlayer / Thread 之间共享
  const [videoDir, setVideoDir] = useQueryState("videoDir", {
    defaultValue: "",
  });

  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <Toaster />
      <ThreadProvider>
        <StreamProvider>
          {/* 三栏布局: TopNav + [LeftSidebar | VideoPlayer | Thread] */}
          {/* 灰色画布 + 三张白色圆角浮层卡片 (NotebookLM 风格) */}
          <div className="flex h-screen w-full flex-col bg-[#F0F2F5]">
            <TopNav videoDir={videoDir} />

            {/* px-4 提供每边 16px 留白; min-[1700px]: 断点卡在两屏之间 (笔记本1536不触发/外接1920触发) */}
            <div className="flex min-h-0 flex-1 px-4 py-3">
              <div className="flex h-full w-full min-h-0 gap-4">
                {/* 左卡: 笔记本 260px / 外接 320px */}
                <div className="flex h-full w-[260px] min-[1700px]:w-[320px] flex-shrink-0 overflow-hidden rounded-xl bg-white shadow-soft-md ring-1 ring-zinc-200/60">
                  <LeftSidebar
                    videoDir={videoDir}
                    onVideoDirChange={setVideoDir}
                  />
                </div>

                {/* 中卡: 方框 (国内视频站风格, 无圆角), flex-1 自然填满 */}
                <div className="flex h-full min-w-0 flex-1 overflow-hidden shadow-soft-md ring-1 ring-black/10">
                  <VideoPlayer
                    videoDir={videoDir}
                    onVideoDirChange={setVideoDir}
                  />
                </div>

                {/* 右卡: 笔记本 380px / 外接 460px */}
                <div className="flex h-full w-[380px] min-[1700px]:w-[460px] flex-shrink-0 overflow-hidden rounded-xl bg-white shadow-soft-md ring-1 ring-zinc-200/60">
                  <Thread />
                </div>
              </div>
            </div>
          </div>
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}
