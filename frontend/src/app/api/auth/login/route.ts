import { NextRequest, NextResponse } from "next/server";
import {
  setAuthCookie,
  signToken,
  verifyCredentials,
} from "@/lib/auth";

export const runtime = "nodejs";

/**
 * 密码登录.
 * 走 lib/auth.ts::verifyCredentials, 该函数在内部完成 phone 查询 + bcrypt 校验,
 * 不把 password_hash 暴露到路由层.
 */
export async function POST(req: NextRequest) {
  let body: { phone?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体格式错误" }, { status: 400 });
  }

  const phone = (body.phone || "").trim();
  const password = body.password || "";

  if (!phone || !password) {
    return NextResponse.json(
      { error: "手机号和密码不能为空" },
      { status: 400 },
    );
  }

  try {
    const user = await verifyCredentials(phone, password);
    // 统一报错避免手机号枚举攻击
    if (!user) {
      return NextResponse.json(
        { error: "手机号或密码错误" },
        { status: 401 },
      );
    }
    const token = await signToken({ sub: user.id, phone: user.phone });
    await setAuthCookie(token);
    return NextResponse.json({ user });
  } catch (e) {
    console.error("[auth/login] failed:", e);
    return NextResponse.json(
      { error: "登录失败, 请检查 Postgres 服务是否正常" },
      { status: 503 },
    );
  }
}
