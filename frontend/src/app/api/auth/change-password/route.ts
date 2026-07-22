import { NextRequest, NextResponse } from "next/server";
import { changePassword, getCurrentUser } from "@/lib/auth";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  // /api/* 不被 middleware 保护, 这里手动校验登录态
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }

  let body: { old_password?: string; new_password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体格式错误" }, { status: 400 });
  }

  const oldP = body.old_password || "";
  const newP = body.new_password || "";

  if (newP.length < 6) {
    return NextResponse.json(
      { error: "新密码至少 6 位" },
      { status: 400 },
    );
  }
  if (oldP === newP) {
    return NextResponse.json(
      { error: "新密码不能和旧密码相同" },
      { status: 400 },
    );
  }

  try {
    const ok = await changePassword(user.id, oldP, newP);
    if (!ok) {
      return NextResponse.json(
        { error: "旧密码错误" },
        { status: 401 },
      );
    }
    return NextResponse.json({ ok: true });
  } catch (e) {
    console.error("[auth/change-password] failed:", e);
    return NextResponse.json(
      { error: "修改失败, 请稍后重试" },
      { status: 500 },
    );
  }
}
