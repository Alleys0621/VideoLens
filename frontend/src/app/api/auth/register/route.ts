import { NextRequest, NextResponse } from "next/server";
import { createUser, getUserByPhone } from "@/lib/auth";

export const runtime = "nodejs";

const PHONE_RE = /^1[3-9]\d{9}$/;

export async function POST(req: NextRequest) {
  let body: {
    phone?: string;
    password?: string;
    display_name?: string;
  };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体格式错误" }, { status: 400 });
  }

  const phone = (body.phone || "").trim();
  const password = body.password || "";

  if (!PHONE_RE.test(phone)) {
    return NextResponse.json({ error: "手机号格式不正确" }, { status: 400 });
  }
  if (password.length < 6) {
    return NextResponse.json({ error: "密码至少 6 位" }, { status: 400 });
  }

  try {
    const existing = await getUserByPhone(phone);
    if (existing) {
      return NextResponse.json({ error: "该手机号已注册" }, { status: 409 });
    }
    // 注册和登录分离: 只创建账号, 不签 token 不设 cookie.
    // 用户需要前往 /login 主动登录.
    const user = await createUser({
      phone,
      password,
      display_name: body.display_name,
    });
    return NextResponse.json({ user });
  } catch (e) {
    console.error("[auth/register] failed:", e);
    return NextResponse.json(
      { error: "注册失败, 请检查后端 LangGraph 服务是否正常" },
      { status: 503 },
    );
  }
}
