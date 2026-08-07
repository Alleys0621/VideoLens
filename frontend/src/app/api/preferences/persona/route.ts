import { NextRequest, NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";
import { query } from "@/lib/db";
import { PERSONAS } from "@/lib/personas";

export const runtime = "nodejs";

export async function GET() {
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "未登录" }, { status: 401 });
  try {
    const { rows } = await query<{ default_persona_id: string }>(
      "SELECT default_persona_id FROM user_preferences WHERE user_id = $1",
      [user.id],
    );
    return NextResponse.json({ persona_id: rows[0]?.default_persona_id ?? "alleys" });
  } catch (e) {
    console.error("[preferences/persona/GET] failed:", e);
    return NextResponse.json({ error: "查询失败" }, { status: 503 });
  }
}

export async function PUT(req: NextRequest) {
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "未登录" }, { status: 401 });
  const body = (await req.json().catch(() => ({}))) as { persona_id?: string };
  if (!body.persona_id || !PERSONAS.some((p) => p.id === body.persona_id)) {
    return NextResponse.json({ error: "无效的人设" }, { status: 400 });
  }
  try {
    await query(
      `INSERT INTO user_preferences (user_id, default_persona_id, updated_at)
       VALUES ($1, $2, now())
       ON CONFLICT (user_id) DO UPDATE
         SET default_persona_id = EXCLUDED.default_persona_id,
             updated_at = now()`,
      [user.id, body.persona_id],
    );
    return NextResponse.json({ ok: true, persona_id: body.persona_id });
  } catch (e) {
    console.error("[preferences/persona/PUT] failed:", e);
    return NextResponse.json({ error: "保存失败" }, { status: 503 });
  }
}
