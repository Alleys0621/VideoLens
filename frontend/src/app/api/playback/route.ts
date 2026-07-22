import { NextRequest, NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { getServerClient } from "@/lib/langgraph-client";

export const runtime = "nodejs";

/**
 * 用户视频播放进度持久化.
 *
 * 存储位置: LangGraph Store, namespace = ["playback", user_id], key = video_dir
 * value: { position, duration, completed, updated_at }
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

interface PlaybackRecord {
  position: number;
  duration: number | null;
  completed: boolean;
  updated_at: string;
  // 索引签名: 满足 putItem 的 Record<string, unknown> 类型
  [key: string]: unknown;
}

function playbackNamespace(userId: string): string[] {
  return ["playback", userId];
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

  const client = getServerClient();
  try {
    const item = await client.store.getItem(playbackNamespace(user.id), videoDir);
    if (!item) {
      return NextResponse.json({ position: null });
    }
    const rec = item.value as PlaybackRecord;
    // 完成的视频从头开始
    if (rec.completed) {
      return NextResponse.json({ position: 0, duration: rec.duration, completed: true });
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

  const rec: PlaybackRecord = {
    // 完成的视频 position 归零, 下次从头开始
    position: completed ? 0 : position,
    duration,
    completed,
    updated_at: new Date().toISOString(),
  };

  const client = getServerClient();
  try {
    await client.store.putItem(playbackNamespace(user.id), videoDir, rec);
    return NextResponse.json({ ok: true, completed });
  } catch (e) {
    console.error("[playback/POST] failed:", e);
    return NextResponse.json({ error: "保存失败" }, { status: 503 });
  }
}
