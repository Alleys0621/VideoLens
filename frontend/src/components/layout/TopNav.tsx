"use client";

import { Settings, Menu } from "lucide-react";
import { UserMenu } from "./UserMenu";

/**
 * TopNav — NotebookLM 风格透明导航栏
 * Logo + 系统名 (Alleys) + 面包屑 (当前视频) + 右侧操作
 * 贴在灰色画布上, 无独立背景/边框
 *
 * iPad 竖屏: 左侧显示 hamburger 菜单按钮 (onToggleSidebar 控制左栏抽屉)
 * 桌面: onToggleSidebar = undefined, 不渲染 hamburger
 */
export function TopNav({
  videoDir,
  onToggleSidebar,
}: {
  videoDir: string;
  onToggleSidebar?: () => void;
}) {
  const crumbs = videoDir ? videoDir.split("/").filter(Boolean) : [];

  return (
    <header className="safe-area-top flex h-14 flex-shrink-0 items-center gap-3 px-3 sm:px-5">
      {/* iPad 竖屏: hamburger 菜单 (44×44 触控热区) */}
      {onToggleSidebar && (
        <button
          onClick={onToggleSidebar}
          className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-lg text-zinc-500 transition-colors hover:bg-white/70 hover:text-zinc-700"
          title="打开剧集列表"
          aria-label="打开剧集列表"
        >
          <Menu className="h-5 w-5" />
        </button>
      )}

      {/* Logo + 系统名 */}
      {/* AlleysVid 品牌 logo + 艺术字标题 */}
      <div className="flex flex-shrink-0 items-center gap-2">
        <img
          src="/alleysvid-logo.png"
          alt="AlleysVid"
          className="h-9 w-auto"
        />
        <span className="alleysvid-brand text-xl">AlleysVid</span>
      </div>

      {/* 分隔线 */}
      <div className="hidden h-5 w-px bg-zinc-300/70 sm:block" />

      {/* 面包屑: 当前视频 (truncate 防止窄屏挤爆) */}
      <nav className="flex min-w-0 flex-1 items-center gap-1.5 text-sm">
        {crumbs.length > 0 ? (
          crumbs.map((seg, i) => (
            <span
              key={i}
              className="flex min-w-0 items-center gap-1.5"
            >
              {i > 0 && <span className="text-zinc-300">/</span>}
              <span
                className={
                  i === crumbs.length - 1
                    ? "truncate font-medium text-zinc-700"
                    : "truncate text-zinc-400"
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

      {/* 右侧操作 */}
      <button
        className="flex h-10 flex-shrink-0 items-center gap-1.5 rounded-lg px-3 text-[13px] text-zinc-500 transition-colors hover:bg-white/70 hover:text-zinc-700"
        title="设置"
      >
        <Settings className="h-4 w-4" />
      </button>
      <UserMenu />
    </header>
  );
}
