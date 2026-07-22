"use client";

import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useMemo, useRef, useState } from "react";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { PanelRightOpen, PanelRightClose, Pin, Pencil, Trash2 } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";

/* ---------- 工具函数 ---------- */

// metadata 自定义字段约定:
//   pinned: boolean        — 是否置顶
//   custom_title: string   — 自定义标题 (空字符串/缺失时回退到首条消息预览)
function isPinned(t: Thread): boolean {
  return !!((t.metadata as Record<string, unknown> | null)?.pinned);
}

function getThreadTitle(t: Thread): string {
  const meta = t.metadata as Record<string, unknown> | null;
  const custom = meta?.custom_title;
  if (typeof custom === "string" && custom.trim()) return custom.trim();
  const values = t.values as { messages?: { content: unknown }[] } | null;
  if (values?.messages?.length) {
    const text = getContentString(values.messages[0].content as never);
    if (text) return text;
  }
  return t.thread_id.slice(0, 8);
}

type GroupKey = "pinned" | "today" | "week" | "month" | "earlier";

function getGroupKey(t: Thread): GroupKey {
  if (isPinned(t)) return "pinned";
  const updated = new Date(t.updated_at).getTime();
  if (Number.isNaN(updated)) return "earlier";
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  if (updated >= startOfToday.getTime()) return "today";
  if (updated >= now - 7 * dayMs) return "week";
  if (updated >= now - 30 * dayMs) return "month";
  return "earlier";
}

const GROUP_LABELS: Record<GroupKey, string> = {
  pinned: "置顶",
  today: "今天",
  week: "7天内",
  month: "30天内",
  earlier: "更早",
};

const GROUP_ORDER: GroupKey[] = ["pinned", "today", "week", "month", "earlier"];

// 把扁平 threads 按 group 分桶, 每桶内按 updated_at 降序
function groupThreads(threads: Thread[]): Record<GroupKey, Thread[]> {
  const buckets: Record<GroupKey, Thread[]> = {
    pinned: [],
    today: [],
    week: [],
    month: [],
    earlier: [],
  };
  for (const t of threads) buckets[getGroupKey(t)].push(t);
  for (const k of GROUP_ORDER) {
    buckets[k].sort(
      (a, b) =>
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    );
  }
  return buckets;
}

/* ---------- 单个 Thread 行 ---------- */
function ThreadItem({
  thread,
  isActive,
  renaming,
  renameValue,
  pendingDelete,
  onRenameChange,
  onRenameCommit,
  onRenameCancel,
  onStartRename,
  onTogglePin,
  onRequestDelete,
  onConfirmDelete,
  onCancelDelete,
  onClick,
}: {
  thread: Thread;
  isActive: boolean;
  renaming: boolean;
  renameValue: string;
  pendingDelete: boolean;
  onRenameChange: (v: string) => void;
  onRenameCommit: () => void;
  onRenameCancel: () => void;
  onStartRename: () => void;
  onTogglePin: () => void;
  onRequestDelete: () => void;
  onConfirmDelete: () => void;
  onCancelDelete: () => void;
  onClick: () => void;
}) {
  // 重命名 input ref — 用于 autoFocus + 选中全文
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (renaming && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [renaming]);

  // 删除二次确认 — 5s 不操作自动取消
  useEffect(() => {
    if (!pendingDelete) return;
    const t = setTimeout(() => onCancelDelete(), 5000);
    return () => clearTimeout(t);
  }, [pendingDelete, onCancelDelete]);

  // --- 删除态: 红色确认 bar ---
  if (pendingDelete) {
    return (
      <div className="flex w-full items-center justify-between gap-2 rounded-lg bg-rose-50 px-2 py-1.5 text-xs text-rose-700 ring-1 ring-rose-200">
        <span className="truncate">确认删除此对话?</span>
        <div className="flex flex-shrink-0 items-center gap-1">
          <button
            onClick={onConfirmDelete}
            className="rounded-md bg-rose-600 px-2 py-1 text-[11px] font-medium text-white transition-colors hover:bg-rose-700"
          >
            删除
          </button>
          <button
            onClick={onCancelDelete}
            className="rounded-md px-2 py-1 text-[11px] font-medium text-rose-600 transition-colors hover:bg-rose-100"
          >
            取消
          </button>
        </div>
      </div>
    );
  }

  // --- 重命名态: 行内 input ---
  if (renaming) {
    return (
      <div className="flex w-full items-center rounded-lg bg-white px-2 py-1 ring-1 ring-indigo-300">
        <input
          ref={inputRef}
          value={renameValue}
          onChange={(e) => onRenameChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              onRenameCommit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              onRenameCancel();
            }
          }}
          onBlur={onRenameCommit}
          placeholder="输入新名称"
          className="w-full bg-transparent py-1.5 text-sm text-zinc-800 outline-none placeholder:text-zinc-400"
        />
      </div>
    );
  }

  // --- 正常态: hover 显示三个图标按钮 ---
  const pinned = isPinned(thread);
  return (
    <div
      className={cn(
        "group/item flex w-full items-center gap-1 rounded-lg px-2 py-1.5 transition-colors",
        isActive
          ? "bg-zinc-100 text-zinc-900"
          : "text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900",
      )}
    >
      <button
        onClick={onClick}
        className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
      >
        {pinned && (
          <Pin className="h-3 w-3 flex-shrink-0 rotate-45 text-amber-500" />
        )}
        <span className="truncate text-sm">
          {getThreadTitle(thread)}
        </span>
      </button>
      <div className="flex flex-shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover/item:opacity-100">
        <IconButton
          title={pinned ? "取消置顶" : "置顶"}
          onClick={onTogglePin}
          active={pinned}
        >
          <Pin className={cn("h-3.5 w-3.5", pinned && "rotate-45")} />
        </IconButton>
        <IconButton title="重命名" onClick={onStartRename}>
          <Pencil className="h-3.5 w-3.5" />
        </IconButton>
        <IconButton title="删除" onClick={onRequestDelete} danger>
          <Trash2 className="h-3.5 w-3.5" />
        </IconButton>
      </div>
    </div>
  );
}

function IconButton({
  children,
  onClick,
  title,
  active,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  active?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={cn(
        "flex h-6 w-6 items-center justify-center rounded-md text-zinc-400 transition-colors",
        danger
          ? "hover:bg-rose-50 hover:text-rose-600"
          : active
            ? "text-amber-500 hover:bg-amber-50 hover:text-amber-600"
            : "hover:bg-zinc-200 hover:text-zinc-700",
      )}
    >
      {children}
    </button>
  );
}

/* ---------- 分组列表 ---------- */
function ThreadList({
  threads,
  onThreadClick,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { deleteThread, updateThreadMetadata } = useThreads();

  // 重命名态
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  // 删除二次确认态
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const groups = useMemo(() => groupThreads(threads), [threads]);

  const commitRename = async (threadId: string) => {
    const newTitle = renameValue.trim();
    setRenamingId(null);
    // 空值: 不更新 (保持原标题)
    if (!newTitle) return;
    // 原标题和新标题相同: 跳过
    const t = threads.find((x) => x.thread_id === threadId);
    if (t && getThreadTitle(t) === newTitle) return;
    try {
      await updateThreadMetadata(threadId, { custom_title: newTitle });
    } catch {
      /* provider 已 toast */
    }
  };

  const confirmDelete = async (tid: string) => {
    setConfirmDeleteId(null);
    try {
      await deleteThread(tid);
    } catch {
      /* provider 已 toast */
    }
  };

  // 删除当前选中 thread 后, 清空 URL 里的 threadId
  const handleConfirmDelete = (tid: string) => {
    confirmDelete(tid).then(() => {
      if (tid === threadId) setThreadId(null);
    });
  };

  const handleTogglePin = async (t: Thread) => {
    try {
      await updateThreadMetadata(t.thread_id, { pinned: !isPinned(t) });
    } catch {
      /* provider 已 toast */
    }
  };

  return (
    <div className="flex h-full w-full flex-col gap-1 overflow-y-scroll px-1 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {GROUP_ORDER.map((key) => {
        const list = groups[key];
        if (list.length === 0) return null;
        return (
          <div key={key} className="flex flex-col gap-0.5">
            <div className="px-2 pb-0.5 pt-3 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
              {GROUP_LABELS[key]}
            </div>
            {list.map((t) => (
              <ThreadItem
                key={t.thread_id}
                thread={t}
                isActive={t.thread_id === threadId}
                renaming={renamingId === t.thread_id}
                renameValue={renameValue}
                pendingDelete={confirmDeleteId === t.thread_id}
                onRenameChange={setRenameValue}
                onRenameCommit={() => commitRename(t.thread_id)}
                onRenameCancel={() => setRenamingId(null)}
                onStartRename={() => {
                  setRenamingId(t.thread_id);
                  setRenameValue(getThreadTitle(t));
                }}
                onTogglePin={() => handleTogglePin(t)}
                onRequestDelete={() => setConfirmDeleteId(t.thread_id)}
                onConfirmDelete={() => handleConfirmDelete(t.thread_id)}
                onCancelDelete={() => setConfirmDeleteId(null)}
                onClick={() => {
                  onThreadClick?.(t.thread_id);
                  if (t.thread_id !== threadId) setThreadId(t.thread_id);
                }}
              />
            ))}
          </div>
        );
      })}
      {threads.length === 0 && (
        <div className="px-3 py-8 text-center text-xs text-zinc-400">
          暂无历史对话
        </div>
      )}
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton key={`skeleton-${i}`} className="h-10 w-[280px]" />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-3 border-r-[1px] border-slate-300 lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-3">
          <h1 className="text-base font-semibold tracking-tight text-zinc-800">
            历史对话
          </h1>
          <Button
            className="hover:bg-gray-100"
            variant="ghost"
            size="sm"
            onClick={() => setChatHistoryOpen((p) => !p)}
          >
            {chatHistoryOpen ? (
              <PanelRightOpen className="size-5" />
            ) : (
              <PanelRightClose className="size-5" />
            )}
          </Button>
        </div>
        {threadsLoading ? (
          <ThreadHistoryLoading />
        ) : (
          <ThreadList threads={threads} />
        )}
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent side="left" className="flex lg:hidden">
            <SheetHeader>
              <SheetTitle>历史对话</SheetTitle>
            </SheetHeader>
            <ThreadList
              threads={threads}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
