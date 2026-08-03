import { NextRequest, NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { query } from "@/lib/db";

export const runtime = "nodejs";

/**
 * 用户实时观看状态上报 (前端播放器定时上报).
 *
 * 存储: Postgres `watching_state` 表 (user_id PK)
 *   video_dir / video_time / is_playing / updated_at
 *
 * POST /api/watching
 *   body: { video_dir, video_time, is_playing? }
 *   → { ok: true }
 *
 * user_id 从 JWT 拿, 不信任前端.
 */

export async function POST(req: NextRequest) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }

  let body: { video_dir?: string; video_time?: number; is_playing?: boolean };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体格式错误" }, { status: 400 });
  }

  const videoDir = body.video_dir;
  const videoTime = body.video_time;
  if (!videoDir || typeof videoTime !== "number" || videoTime < 0) {
    return NextResponse.json({ error: "参数错误" }, { status: 400 });
  }
  const isPlaying = body.is_playing !== false;

  try {
    await query(
      `INSERT INTO watching_state (user_id, video_dir, video_time, is_playing, updated_at)
       VALUES ($1, $2, $3, $4, now())
       ON CONFLICT (user_id)
       DO UPDATE SET
         video_dir  = EXCLUDED.video_dir,
         video_time = EXCLUDED.video_time,
         is_playing = EXCLUDED.is_playing,
         updated_at = now()`,
      [user.id, videoDir, videoTime, isPlaying],
    );
    return NextResponse.json({ ok: true });
  } catch (e) {
    console.error("[watching/POST] failed:", e);
    return NextResponse.json({ error: "保存失败" }, { status: 503 });
  }
}
