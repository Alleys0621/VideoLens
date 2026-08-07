import { NextRequest, NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { query } from "@/lib/db";
import { PERSONAS } from "@/lib/personas";

export const runtime = "nodejs";

/**
 * PATCH /api/chat-threads/:id
 *   body: { custom_title?, pinned? }
 * DELETE /api/chat-threads/:id
 *   先调 LangGraph /threads/{id} DELETE 删 state, 再删 PG threads 行.
 */

interface ThreadRow {
  thread_id: string;
  custom_title: string | null;
  pinned: boolean;
  persona_id: string;
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
      custom_title: row.custom_title,
      pinned: row.pinned,
      persona_id: row.persona_id,
    },
    values: null,
  };
}

async function requireOwnership(threadId: string, userId: string) {
  const { rows } = await query<{ thread_id: string }>(
    "SELECT thread_id FROM threads WHERE thread_id = $1 AND user_id = $2",
    [threadId, userId],
  );
  return rows.length > 0;
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }
  const { id } = await params;

  if (!(await requireOwnership(id, user.id))) {
    return NextResponse.json({ error: "无权操作此会话" }, { status: 403 });
  }

  let body: { custom_title?: string; pinned?: boolean; persona_id?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体格式错误" }, { status: 400 });
  }

  const sets: string[] = [];
  const vals: unknown[] = [];
  // $1=user_id, 字段从 $2 开始, 最后一个是 thread_id
  if (typeof body.custom_title === "string") {
    vals.push(body.custom_title.slice(0, 200));
    sets.push(`custom_title = $${vals.length + 1}`);
  }
  if (typeof body.pinned === "boolean") {
    vals.push(body.pinned);
    sets.push(`pinned = $${vals.length + 1}`);
  }
  if (typeof body.persona_id === "string") {
    if (!PERSONAS.some((p) => p.id === body.persona_id)) {
      return NextResponse.json({ error: "无效的人设" }, { status: 400 });
    }
    vals.push(body.persona_id);
    sets.push(`persona_id = $${vals.length + 1}`);
  }
  if (sets.length === 0) {
    return NextResponse.json({ error: "没有可更新字段" }, { status: 400 });
  }

  const queryParams: unknown[] = [user.id, ...vals, id];
  try {
    const { rows } = await query<ThreadRow>(
      `UPDATE threads SET ${sets.join(", ")}
       WHERE thread_id = $${queryParams.length} AND user_id = $1
       RETURNING thread_id, custom_title, pinned, persona_id, created_at, updated_at`,
      queryParams,
    );
    if (rows.length === 0) {
      return NextResponse.json({ error: "无权操作此会话" }, { status: 403 });
    }
    return NextResponse.json({ ok: true, thread: rowToThread(rows[0]) });
  } catch (e) {
    console.error("[chat-threads/PATCH] failed:", e);
    return NextResponse.json({ error: "更新失败" }, { status: 503 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }
  const { id } = await params;

  if (!(await requireOwnership(id, user.id))) {
    return NextResponse.json({ error: "无权操作此会话" }, { status: 403 });
  }

  // 1. 先删 LangGraph 那边的 state (checkpoints)
  try {
    const lgUrl = process.env.LANGGRAPH_API_URL || "http://127.0.0.1:2024";
    await fetch(`${lgUrl}/threads/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
  } catch (e) {
    console.error("[chat-threads/DELETE] langgraph delete failed:", e);
  }

  // 2. 再删 PG 元数据 (CASCADE 会带走 potential 关联, 这里只有 threads 自身)
  try {
    await query("DELETE FROM threads WHERE thread_id = $1 AND user_id = $2", [
      id,
      user.id,
    ]);
    return NextResponse.json({ ok: true });
  } catch (e) {
    console.error("[chat-threads/DELETE] pg delete failed:", e);
    return NextResponse.json({ error: "删除失败" }, { status: 503 });
  }
}
