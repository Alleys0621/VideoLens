"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { Toaster } from "@/components/ui/sonner";

export interface AuthUser {
  id: string;
  phone: string;
  display_name: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean; // 初次加载中 (mount → /me 返回前为 true)
  login: (phone: string, password: string) => Promise<AuthUser>;
  register: (input: {
    phone: string;
    password: string;
    display_name?: string;
  }) => Promise<AuthUser>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

async function fetchJson(url: string, init?: RequestInit) {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error((data as { error?: string }).error || "请求失败");
  }
  return data;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = (await fetchJson("/api/auth/me")) as { user: AuthUser | null };
      setUser(data.user);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const login = useCallback(
    async (phone: string, password: string): Promise<AuthUser> => {
      const data = (await fetchJson("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ phone, password }),
      })) as { user: AuthUser };
      setUser(data.user);
      return data.user;
    },
    [],
  );

  const register = useCallback(
    async (input: {
      phone: string;
      password: string;
      display_name?: string;
    }): Promise<AuthUser> => {
      const data = (await fetchJson("/api/auth/register", {
        method: "POST",
        body: JSON.stringify(input),
      })) as { user: AuthUser };
      // 注册 ≠ 登录: 后端不再 setAuthCookie, 这里也不修改 user state.
      // 调用方应在成功后跳转到 /login.
      return data.user;
    },
    [],
  );

  const logout = useCallback(async () => {
    await fetchJson("/api/auth/logout", { method: "POST" });
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{ user, loading, login, register, logout, refresh }}
    >
      {/* Toaster 挂在 client provider 内, 避免 SSR 调用 useTheme */}
      <Toaster />
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
