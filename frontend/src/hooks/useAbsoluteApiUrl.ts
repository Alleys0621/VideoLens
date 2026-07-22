import { useEffect, useState } from "react";

/**
 * 把相对 URL 转成绝对 URL。
 * 原因: langgraph SDK 内部用 `new URL(apiUrl)` 解析 URL，相对路径会抛 TypeError。
 * 当 NEXT_PUBLIC_API_URL=/api (相对, 让浏览器走 passthrough) 时必须先转绝对。
 *
 * 行为:
 * - SSR 阶段返回 null (避免 hydration mismatch: 服务端没 window)
 * - 客户端 mount 后返回真实绝对 URL
 *   · 入参已经是 http(s):// 开头 → 原样返回
 *   · 入参是 /api 这种相对路径 → prepend window.location.origin
 */
export function useAbsoluteApiUrl(apiUrl: string | undefined): string | null {
  const [absolute, setAbsolute] = useState<string | null>(null);

  useEffect(() => {
    if (!apiUrl) {
      setAbsolute(null);
      return;
    }
    if (/^https?:\/\//i.test(apiUrl)) {
      setAbsolute(apiUrl);
    } else {
      const path = apiUrl.startsWith("/") ? apiUrl : `/${apiUrl}`;
      setAbsolute(`${window.location.origin}${path}`);
    }
  }, [apiUrl]);

  return absolute;
}
