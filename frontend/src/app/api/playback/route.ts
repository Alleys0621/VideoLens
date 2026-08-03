import { NextRequest, NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { query } from "@/lib/db";

export const runtime = "nodejs";

/**
 * 用户视频播放进度持久化.
 *
 * 存储位置: Postgres `playback_progress` 表
 *   (user_id, video_dir) 联合主键
 *   position / duration / completed / updated_at
 *
 * GET /api/playback?video_dir=xxx
 *   → { position, duration, completed, updated_at } 或 { position: null }
 *
 * POST /api/playback
 *   body: { video_dir, position, duration?, completed? }
 *   → { ok: true }
 *
 * user_id 从 JWT 拿 (HttpOnly cookie), 不信任前端传的 user_id.
 */

interface PlaybackRow {
  position: number;
  duration: number | null;
  completed: boolean;
  updated_at: string;
}

export async function GET(req: NextRequest) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }

  const videoDir = req.nextUrl.searchParams.get("video_dir");
  if (!videoDir) {
    return NextResponse.json({ error: "缺 video_dir 参数" }, { status: 400 });
  }

  try {
    const { rows } = await query<PlaybackRow>(
      `SELECT position, duration, completed, updated_at
       FROM playback_progress
       WHERE user_id = $1 AND video_dir = $2`,
      [user.id, videoDir],
    );
    if (rows.length === 0) {
      return NextResponse.json({ position: null });
    }
    const rec = rows[0];
    // 完成的视频从头开始
    if (rec.completed) {
      return NextResponse.json({
        position: 0,
        duration: rec.duration,
        completed: true,
      });
    }
    return NextResponse.json({
      position: rec.position,
      duration: rec.duration,
      completed: false,
      updated_at: rec.updated_at,
    });
  } catch (e) {
    console.error("[playback/GET] failed:", e);
    return NextResponse.json({ error: "查询失败" }, { status: 503 });
  }
}

export async function POST(req: NextRequest) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }

  let body: {
    video_dir?: string;
    position?: number;
    duration?: number;
    completed?: boolean;
  };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体格式错误" }, { status: 400 });
  }

  const videoDir = body.video_dir;
  const position = body.position;
  if (!videoDir || typeof position !== "number" || position < 0) {
    return NextResponse.json({ error: "参数错误" }, { status: 400 });
  }

  const duration = typeof body.duration === "number" ? body.duration : null;
  // 完成判断: 前端显式传 completed, 或 duration 已知且 position >= duration * 0.95
  let completed = !!body.completed;
  if (!completed && duration && duration > 0 && position >= duration * 0.95) {
    completed = true;
  }

  // 完成的视频 position 归零, 下次从头开始
  const finalPosition = completed ? 0 : position;

  try {
    // UPSERT (user_id, video_dir) 联合主键
    await query(
      `INSERT INTO playback_progress (user_id, video_dir, position, duration, completed, updated_at)
       VALUES ($1, $2, $3, $4, $5, now())
       ON CONFLICT (user_id, video_dir)
       DO UPDATE SET
         position   = EXCLUDED.position,
         duration   = EXCLUDED.duration,
         completed  = EXCLUDED.completed,
         updated_at = now()`,
      [user.id, videoDir, finalPosition, duration, completed],
    );

    // 看完一集的记忆沉淀 hook (空实现, 后续填 episodic memory 总结逻辑)
    if (completed) {
      // TODO: 异步 LLM 总结本集对话 → 存 episodic memory 表
      console.log(`[playback] completed: user=${user.id} video=${videoDir}`);
    }
    return NextResponse.json({ ok: true, completed });
  } catch (e) {
    console.error("[playback/POST] failed:", e);
    return NextResponse.json({ error: "保存失败" }, { status: 503 });
  }
}
