import { validate } from "uuid";
import { getApiKey } from "@/lib/api-key";
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
import { createClient } from "./client";
import { useAbsoluteApiUrl } from "@/hooks/useAbsoluteApiUrl";
import { toast } from "sonner";
import { useAuth } from "./Auth";

interface ThreadContextType {
  getThreads: () => Promise<Thread[]>;
  threads: Thread[];
  setThreads: Dispatch<SetStateAction<Thread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
  /** 删除 thread：乐观从本地移除，失败回滚 + toast */
  deleteThread: (threadId: string) => Promise<void>;
  /** 合并式更新 thread.metadata：乐观更新本地，失败回滚 + toast。返回更新后的 thread 或 null */
  updateThreadMetadata: (
    threadId: string,
    metadata: Record<string, unknown>,
  ) => Promise<Thread | null>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const envAuthScheme: string | undefined = process.env.NEXT_PUBLIC_AUTH_SCHEME;

  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId] = useQueryState("assistantId");
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: envAuthScheme || "",
  });
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);

  // 当前登录用户 — thread 按 user_id 隔离
  const { user } = useAuth();

  // 相对路径 (/api) 转绝对 — langgraph SDK new URL() 不支持相对路径
  const absoluteApiUrl = useAbsoluteApiUrl(apiUrl);

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    const resolvedAssistantId = assistantId || envAssistantId;
    if (!absoluteApiUrl || !resolvedAssistantId) return [];
    // 未登录不拉历史 (避免拉到匿名 thread 或越权)
    if (!user) return [];
    const client = createClient(
      absoluteApiUrl,
      getApiKey() ?? undefined,
      authScheme || undefined,
    );

    const threads = await client.threads.search({
      metadata: {
        ...getThreadSearchMetadata(resolvedAssistantId),
        // 关键: 按 user_id 过滤, SDK 文档 "Exact match for each key/value" → AND 关系
        user_id: user.id,
      },
      limit: 100,
    });

    return threads;
  }, [absoluteApiUrl, assistantId, authScheme, envAssistantId, user]);

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

  // 删除 thread：乐观从本地移除，失败回滚 + toast
  const deleteThread = useCallback(
    async (threadId: string): Promise<void> => {
      if (!absoluteApiUrl) return;
      const client = createClient(
        absoluteApiUrl,
        getApiKey() ?? undefined,
        authScheme || undefined,
      );
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId));
      try {
        await client.threads.delete(threadId);
      } catch (e) {
        console.error("deleteThread failed:", e);
        toast.error("删除对话失败，已恢复");
        // 回滚：重拉一次列表
        getThreads().then(setThreads).catch(console.error);
        throw e;
      }
    },
    [absoluteApiUrl, authScheme, getThreads],
  );

  // 合并式更新 metadata：乐观更新本地，失败回滚 + toast
  const updateThreadMetadata = useCallback(
    async (
      threadId: string,
      metadata: Record<string, unknown>,
    ): Promise<Thread | null> => {
      if (!absoluteApiUrl) return null;
      const client = createClient(
        absoluteApiUrl,
        getApiKey() ?? undefined,
        authScheme || undefined,
      );
      let snapshot: Thread | undefined;
      setThreads((prev) =>
        prev.map((t) => {
          if (t.thread_id === threadId) {
            snapshot = t;
            return { ...t, metadata: { ...t.metadata, ...metadata } };
          }
          return t;
        }),
      );
      try {
        const updated = await client.threads.update(threadId, {
          metadata: { ...snapshot?.metadata, ...metadata },
        });
        setThreads((prev) =>
          prev.map((t) => (t.thread_id === threadId ? updated : t)),
        );
        return updated;
      } catch (e) {
        console.error("updateThreadMetadata failed:", e);
        toast.error("更新对话失败，已恢复");
        if (snapshot) {
          setThreads((prev) =>
            prev.map((t) => (t.thread_id === threadId ? snapshot! : t)),
          );
        }
        throw e;
      }
    },
    [absoluteApiUrl, authScheme],
  );

  const value = {
    getThreads,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
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
