import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useEffect,
  useState,
  Dispatch,
  SetStateAction,
} from "react";
import { toast } from "sonner";
import { useAuth } from "./Auth";

/**
 * Thread 元数据 provider (Postgres 后端).
 *
 * 数据流:
 *   - getThreads: GET /api/threads (查 PG threads 表)
 *   - syncThread: POST /api/threads/sync (在 SDK onThreadId 时调用, 把新 thread 写到 PG)
 *   - deleteThread: DELETE /api/threads/:id (PG + LangGraph state 双删)
 *   - updateThreadMetadata: PATCH /api/threads/:id (custom_title / pinned)
 *
 * thread state (messages) 不在这里管, 由 useStream 在切换 thread 时拉取.
 */

interface ThreadLike {
  thread_id: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown> | null;
  values: unknown;
}

interface ThreadContextType {
  getThreads: () => Promise<Thread[]>;
  threads: Thread[];
  setThreads: Dispatch<SetStateAction<Thread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
  /** 把 LangGraph 创建的 thread 同步到 PG (StreamProvider 在 onThreadId 时调) */
  syncThread: (threadId: string) => Promise<void>;
  /** 删除 thread：乐观从本地移除，失败回滚 + toast */
  deleteThread: (threadId: string) => Promise<void>;
  /** 合并式更新 thread.metadata：乐观更新本地，失败回滚 + toast */
  updateThreadMetadata: (
    threadId: string,
    metadata: Record<string, unknown>,
  ) => Promise<Thread | null>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

// /api/threads 返回的形状 (兼容 LangGraph SDK Thread 字段)
interface ApiThread {
  thread_id: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown> | null;
  values: unknown;
}

function toSdkThread(t: ApiThread): Thread {
  // 强制转成 SDK Thread 类型 (history/index.tsx 等组件按这个形状访问)
  return t as unknown as Thread;
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const { user } = useAuth();

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    if (!user) return [];
    try {
      const res = await fetch("/api/chat-threads", { cache: "no-store" });
      if (!res.ok) {
        console.error("[threads] GET failed:", res.status);
        return [];
      }
      const data = (await res.json()) as { threads: ApiThread[] };
      return data.threads.map(toSdkThread);
    } catch (e) {
      console.error("[threads] GET error:", e);
      return [];
    }
  }, [user]);

  // user 变化时 (登录/登出/切换账号) 自动重拉 threads
  useEffect(() => {
    if (!user) {
      setThreads([]);
      setThreadsLoading(false);
      return;
    }
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, [user, getThreads]);

  const syncThread = useCallback(
    async (threadId: string): Promise<void> => {
      try {
        const res = await fetch("/api/chat-threads/sync", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ thread_id: threadId }),
        });
        if (!res.ok) {
          console.error("[threads] sync failed:", res.status);
          return;
        }
        const data = (await res.json()) as { thread: ApiThread; ok: boolean };
        if (data.thread) {
          // 合并到本地列表 (去重)
          setThreads((prev) => {
            const exists = prev.find((t) => t.thread_id === threadId);
            if (exists) return prev;
            return [toSdkThread(data.thread), ...prev];
          });
        }
      } catch (e) {
        console.error("[threads] sync error:", e);
      }
    },
    [],
  );

  // 删除 thread：乐观从本地移除，失败回滚 + toast
  const deleteThread = useCallback(
    async (threadId: string): Promise<void> => {
      const snapshot = threads;
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId));
      try {
        const res = await fetch(`/api/chat-threads/${encodeURIComponent(threadId)}`, {
          method: "DELETE",
        });
        if (!res.ok) {
          const data = (await res.json().catch(() => ({}))) as { error?: string };
          throw new Error(data.error || `HTTP ${res.status}`);
        }
      } catch (e) {
        console.error("[threads] delete failed:", e);
        toast.error("删除对话失败，已恢复");
        setThreads(snapshot); // 回滚
        throw e;
      }
    },
    [threads],
  );

  // 合并式更新 metadata：乐观更新本地，失败回滚 + toast
  const updateThreadMetadata = useCallback(
    async (
      threadId: string,
      metadata: Record<string, unknown>,
    ): Promise<Thread | null> => {
      const snapshot = threads.find((t) => t.thread_id === threadId);
      // 乐观更新
      setThreads((prev) =>
        prev.map((t) =>
          t.thread_id === threadId
            ? { ...t, metadata: { ...(t.metadata || {}), ...metadata } }
            : t,
        ),
      );
      try {
        const res = await fetch(`/api/chat-threads/${encodeURIComponent(threadId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(metadata),
        });
        if (!res.ok) {
          const data = (await res.json().catch(() => ({}))) as { error?: string };
          throw new Error(data.error || `HTTP ${res.status}`);
        }
        const data = (await res.json()) as { thread: ApiThread; ok: boolean };
        if (data.thread) {
          const updated = toSdkThread(data.thread);
          setThreads((prev) =>
            prev.map((t) => (t.thread_id === threadId ? updated : t)),
          );
          return updated;
        }
        return null;
      } catch (e) {
        console.error("[threads] update failed:", e);
        toast.error("更新对话失败，已恢复");
        if (snapshot) {
          setThreads((prev) =>
            prev.map((t) => (t.thread_id === threadId ? snapshot : t)),
          );
        }
        throw e;
      }
    },
    [threads],
  );

  const value = {
    getThreads,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
    syncThread,
    deleteThread,
    updateThreadMetadata,
  };

  return (
    <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}

// 防止 nuqs 被误删 import (保留兼容性, 实际未使用)
void useQueryState;
