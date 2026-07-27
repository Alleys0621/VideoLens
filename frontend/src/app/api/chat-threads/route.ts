import { NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { query } from "@/lib/db";

export const runtime = "nodejs";

/**
 * 会话(thread)元数据 CRUD — Postgres 端.
 *
 * 路径特意放在 /api/chat-threads 而不是 /api/threads:
 *   /api/[..._path]/route.ts 是 LangGraph SDK 的透传代理, 会处理 SDK 的
 *   POST /api/threads (创建 thread) 等请求. 若我们把自定义路由放在
 *   /api/threads, Next.js 会用它覆盖代理 → SDK 创建 thread 时拿到 405.
 *   因此自定义元数据接口挪到 /api/chat-threads, 让 /api/threads 走代理.
 *
 * GET /api/chat-threads
 *   → { threads: [{ thread_id, created_at, updated_at, metadata }] }
 */

interface ThreadRow {
  thread_id: string;
  custom_title: string | null;
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

function rowToThread(row: ThreadRow) {
  return {
    thread_id: row.thread_id,
    created_at: row.created_at,
    updated_at: typeof row.updated_at === "string"
      ? row.updated_at
      : new Date(row.updated_at as unknown as Date).toISOString(),
    metadata: {
      user_id: undefined,
      custom_title: row.custom_title,
      pinned: row.pinned,
    },
    values: null,
  };
}

export async function GET() {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }

  try {
    const { rows } = await query<ThreadRow>(
      `SELECT thread_id, custom_title, pinned, created_at, updated_at
       FROM threads
       WHERE user_id = $1
       ORDER BY pinned DESC, updated_at DESC`,
      [user.id],
    );
    return NextResponse.json({ threads: rows.map(rowToThread) });
  } catch (e) {
    console.error("[chat-threads/GET] failed:", e);
    return NextResponse.json({ error: "查询失败" }, { status: 503 });
  }
}
