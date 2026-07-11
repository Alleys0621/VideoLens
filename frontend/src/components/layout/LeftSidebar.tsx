"use client";

import { useEffect, useState } from "react";
import {
  Film,
  Users,
  BookOpen,
  Plus,
  Clock,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

/* ---------- Types (matches /api/videos/list) ---------- */
type EpisodeT = { dir: string; label: string };
type SeasonT = { name: string; episodes: EpisodeT[] };
type ShowT = {
  name: string;
  seasons: SeasonT[];
  directEpisodes: EpisodeT[];
};

/* ---------- Props ---------- */
export interface LeftSidebarProps {
  videoDir: string;
  onVideoDirChange: (dir: string) => void;
}

/* ---------- Tab definitions ---------- */
const TABS = [
  { id: "episodes", label: "剧集", icon: Film },
  { id: "characters", label: "角色", icon: Users },
  { id: "notes", label: "笔记", icon: BookOpen },
] as const;

type TabId = (typeof TABS)[number]["id"];

/* ---------- Component ---------- */
export function LeftSidebar({ videoDir, onVideoDirChange }: LeftSidebarProps) {
  const [activeTab, setActiveTab] = useState<TabId>("episodes");
  const [shows, setShows] = useState<ShowT[]>([]);
  const [expandedShow, setExpandedShow] = useState<string | null>(null);
  const [expandedSeason, setExpandedSeason] = useState<string | null>(null);

  // 获取视频列表
  useEffect(() => {
    fetch("/api/videos/list")
      .then((r) => r.json())
      .then((data: ShowT[]) => {
        setShows(data);
        // 自动展开当前选中剧集所属的作品/季
        if (data.length && !expandedShow) {
          setExpandedShow(data[0].name);
          if (data[0].seasons.length > 0) {
            setExpandedSeason(data[0].seasons[0].name);
          }
        }
      })
      .catch(() => setShows([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 当 videoDir 变化时，自动展开对应的 show/season
  useEffect(() => {
    if (!videoDir) return;
    for (const show of shows) {
      // 检查 directEpisodes
      for (const ep of show.directEpisodes) {
        if (ep.dir === videoDir) {
          setExpandedShow(show.name);
          return;
        }
      }
      // 检查 seasons
      for (const season of show.seasons) {
        for (const ep of season.episodes) {
          if (ep.dir === videoDir) {
            setExpandedShow(show.name);
            setExpandedSeason(season.name);
            return;
          }
        }
      }
    }
  }, [videoDir, shows]);

  return (
    <aside className="flex w-full flex-col overflow-hidden">
      {/* Tab 栏 */}
      <div className="flex border-b">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex flex-1 items-center justify-center gap-1.5 border-b-2 py-2.5 text-xs font-medium transition-colors",
                active
                  ? "border-indigo-500 text-zinc-900"
                  : "border-transparent text-zinc-400 hover:text-zinc-600",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab 内容 */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {activeTab === "episodes" && (
          <EpisodeList
            shows={shows}
            videoDir={videoDir}
            expandedShow={expandedShow}
            expandedSeason={expandedSeason}
            onToggleShow={(name) =>
              setExpandedShow((prev) => (prev === name ? null : name))
            }
            onToggleSeason={(name) =>
              setExpandedSeason((prev) => (prev === name ? null : name))
            }
            onSelectEpisode={onVideoDirChange}
          />
        )}

        {activeTab === "characters" && (
          <div className="p-4 text-center text-xs text-zinc-400">
            <Users className="mx-auto mb-2 h-8 w-8 text-zinc-300" />
            <p>角色信息将在后续版本中展示</p>
            <p className="mt-1 text-zinc-300">
              需要后端提供角色数据接口
            </p>
          </div>
        )}

        {activeTab === "notes" && (
          <div className="flex flex-col items-center p-6 text-center">
            <BookOpen className="mb-3 h-8 w-8 text-zinc-300" />
            <p className="text-xs text-zinc-400">
              观看过程中可以记录笔记...
            </p>
            <button className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-500 shadow-soft transition-colors hover:bg-zinc-50">
              <Plus className="h-3 w-3" />
              添加笔记
            </button>
          </div>
        )}
      </div>

      {/* 底部：本集摘要 */}
      {videoDir && (
        <div className="border-t bg-zinc-50/50 p-3.5">
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
            本集摘要
          </p>
          <p className="text-xs leading-relaxed text-zinc-500">
            {videoDir.split("/").pop() || "暂无摘要"}
            {shows.length > 0 && " — 剧情摘要数据接入后将在此处展示"}
          </p>
        </div>
      )}
    </aside>
  );
}

/* ---------- EpisodeList sub-component ---------- */
function EpisodeList({
  shows,
  videoDir,
  expandedShow,
  expandedSeason,
  onToggleShow,
  onToggleSeason,
  onSelectEpisode,
}: {
  shows: ShowT[];
  videoDir: string;
  expandedShow: string | null;
  expandedSeason: string | null;
  onToggleShow: (name: string) => void;
  onToggleSeason: (name: string) => void;
  onSelectEpisode: (dir: string) => void;
}) {
  if (shows.length === 0) {
    return (
      <div className="p-6 text-center text-xs text-zinc-400">
        <Film className="mx-auto mb-2 h-8 w-8 text-zinc-300" />
        <p>暂无已处理的视频</p>
        <p className="mt-1 text-zinc-300">
          请先运行 pipeline 处理视频
        </p>
      </div>
    );
  }

  return (
    <div className="p-2">
      {shows.map((show) => {
        const isShowExpanded = expandedShow === show.name;

        return (
          <div key={show.name} className="mb-1">
            {/* 作品名 */}
            <button
              onClick={() => onToggleShow(show.name)}
              className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50"
            >
              <ChevronRight
                className={cn(
                  "h-3.5 w-3.5 flex-shrink-0 text-zinc-400 transition-transform",
                  isShowExpanded && "rotate-90",
                )}
              />
              <span className="truncate">{show.name}</span>
              <span className="ml-auto text-[10px] text-zinc-400">
                {show.seasons.reduce(
                  (acc, s) => acc + s.episodes.length,
                  show.directEpisodes.length,
                )}
                集
              </span>
            </button>

            {/* 季 & 集 */}
            {isShowExpanded && (
              <div className="ml-3 border-l border-zinc-200 pl-2">
                {/* 直接集 (无季) */}
                {show.directEpisodes.length > 0 && (
                  <div className="space-y-0.5">
                    {show.directEpisodes.map((ep) => (
                      <EpisodeItem
                        key={ep.dir}
                        episode={ep}
                        isActive={videoDir === ep.dir}
                        onSelect={() => onSelectEpisode(ep.dir)}
                      />
                    ))}
                  </div>
                )}

                {/* 按季分组 */}
                {show.seasons.map((season) => {
                  const isSeasonExpanded = expandedSeason === season.name;
                  return (
                    <div key={season.name} className="mt-1">
                      <button
                        onClick={() => onToggleSeason(season.name)}
                        className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs font-medium text-zinc-500 transition-colors hover:bg-zinc-50"
                      >
                        <ChevronRight
                          className={cn(
                            "h-3 w-3 flex-shrink-0 text-zinc-400 transition-transform",
                            isSeasonExpanded && "rotate-90",
                          )}
                        />
                        {season.name}
                        <span className="ml-auto text-[10px] text-zinc-400">
                          {season.episodes.length}集
                        </span>
                      </button>

                      {isSeasonExpanded && (
                        <div className="ml-4 space-y-0.5">
                          {season.episodes.map((ep) => (
                            <EpisodeItem
                              key={ep.dir}
                              episode={ep}
                              isActive={videoDir === ep.dir}
                              onSelect={() => onSelectEpisode(ep.dir)}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ---------- EpisodeItem ---------- */
function EpisodeItem({
  episode,
  isActive,
  onSelect,
}: {
  episode: EpisodeT;
  isActive: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "group flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-xs transition-all",
        isActive
          ? "border border-zinc-200 bg-zinc-50 shadow-soft"
          : "border border-transparent text-zinc-500 hover:bg-zinc-50 hover:text-zinc-700",
      )}
    >
      <span className="flex-1 truncate">{episode.label}</span>
      {isActive && (
        <span className="flex items-center gap-1 text-[10px] text-emerald-500">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          播放中
        </span>
      )}
    </button>
  );
}
