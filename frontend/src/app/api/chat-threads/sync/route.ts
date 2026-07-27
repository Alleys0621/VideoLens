import { NextRequest, NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { query } from "@/lib/db";

export const runtime = "nodejs";

/**
 * POST /api/chat-threads/sync
 * body: { thread_id }
 *
 * 把一个 LangGraph 已创建的 thread 同步到 PG threads 表 (幂等 upsert).
 * 前端 StreamProvider 在 SDK onThreadId(id) 回调里调用此接口.
 */
export async function POST(req: NextRequest) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }

  let body: { thread_id?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体格式错误" }, { status: 400 });
  }

  const threadId = body.thread_id;
  if (!threadId) {
    return NextResponse.json({ error: "缺 thread_id" }, { status: 400 });
  }

  try {
    const { rows } = await query<{ thread_id: string; created_at: string; updated_at: string }>(
      `INSERT INTO threads (thread_id, user_id)
       VALUES ($1, $2)
       ON CONFLICT (thread_id) DO NOTHING
       RETURNING thread_id, created_at, updated_at`,
      [threadId, user.id],
    );

    if (rows.length > 0) {
      return NextResponse.json({
        ok: true,
        thread: {
          thread_id: rows[0].thread_id,
          created_at: rows[0].created_at,
          updated_at: rows[0].updated_at,
          metadata: { custom_title: null, pinned: false },
          values: null,
        },
      });
    }

    const { rows: existing } = await query<{
      thread_id: string;
      custom_title: string | null;
      pinned: boolean;
      created_at: string;
      updated_at: string;
    }>(
      `SELECT thread_id, custom_title, pinned, created_at, updated_at
       FROM threads WHERE thread_id = $1 AND user_id = $2`,
      [threadId, user.id],
    );
    if (existing.length === 0) {
      return NextResponse.json({ error: "无权操作此会话" }, { status: 403 });
    }
    const row = existing[0];
    return NextResponse.json({
      ok: true,
      thread: {
        thread_id: row.thread_id,
        created_at: row.created_at,
        updated_at: row.updated_at,
        metadata: { custom_title: row.custom_title, pinned: row.pinned },
        values: null,
      },
    });
  } catch (e) {
    console.error("[chat-threads/sync] failed:", e);
    return NextResponse.json({ error: "同步失败" }, { status: 503 });
  }
}
