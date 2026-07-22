import { Client } from "@langchain/langgraph-sdk";

/**
 * Server-side LangGraph client singleton.
 *
 * 前端 client (providers/client.ts) 走用户态请求 (浏览器 → Next.js → LangGraph).
 * 这里的 server client 直接在 Next.js API Route 内部用, 跳过浏览器,
 * 用服务账号直连 LangGraph Store 做用户数据 CRUD.
 *
 * 读环境变量:
 *   LANGGRAPH_API_URL       — 必填, 默认 http://localhost:2024
 *   LANGSMITH_API_KEY       — 可选, 本地 langgraph dev 不需要
 *   LANGGRAPH_AUTH_SCHEME   — 可选, 例如 langsmith-api-key
 */
let _client: Client | null = null;

export function getServerClient(): Client {
  if (_client) return _client;
  const apiUrl =
    process.env.LANGGRAPH_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:2024";
  const apiKey = process.env.LANGSMITH_API_KEY || undefined;
  const authScheme = process.env.LANGGRAPH_AUTH_SCHEME || undefined;

  _client = new Client({
    apiUrl,
    ...(apiKey && { apiKey }),
    ...(authScheme && {
      defaultHeaders: { "X-Auth-Scheme": authScheme },
    }),
  });
  return _client;
}
