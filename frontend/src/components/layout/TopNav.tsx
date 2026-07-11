"use client";

import { Settings, User } from "lucide-react";

/**
 * TopNav — NotebookLM 风格透明导航栏
 * Logo + 系统名 (Alleys) + 面包屑 (当前视频) + 右侧操作
 * 贴在灰色画布上, 无独立背景/边框
 */
export function TopNav({ videoDir }: { videoDir: string }) {
  const crumbs = videoDir ? videoDir.split("/").filter(Boolean) : [];

  return (
    <header className="flex h-14 flex-shrink-0 items-center gap-4 px-5">
      {/* Logo + 系统名 */}
      {/* AlleysVid 品牌 logo + 艺术字标题 */}
      <div className="flex items-center gap-2">
        <img
          src="/alleysvid-logo.png"
          alt="AlleysVid"
          className="h-9 w-auto"
        />
        <span className="alleysvid-brand text-xl">AlleysVid</span>
      </div>

      {/* 分隔线 */}
      <div className="h-5 w-px bg-zinc-300/70" />

      {/* 面包屑: 当前视频 */}
      <nav className="flex items-center gap-1.5 text-sm">
        {crumbs.length > 0 ? (
          crumbs.map((seg, i) => (
            <span key={i} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-zinc-300">/</span>}
              <span
                className={
                  i === crumbs.length - 1
                    ? "font-medium text-zinc-700"
                    : "text-zinc-400"
                }
              >
                {seg}
              </span>
            </span>
          ))
        ) : (
          <span className="text-zinc-400">选择一集开始陪看</span>
        )}
      </nav>

      <div className="flex-1" />

      {/* 右侧操作 */}
      <button
        className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] text-zinc-500 transition-colors hover:bg-white/70 hover:text-zinc-700"
        title="设置"
      >
        <Settings className="h-4 w-4" />
      </button>
      <button
        className="flex h-8 w-8 items-center justify-center rounded-full bg-white/80 text-zinc-400 ring-1 ring-zinc-200/80 transition-colors hover:bg-white hover:text-zinc-600"
        title="账户"
      >
        <User className="h-4 w-4" />
      </button>
    </header>
  );
}
