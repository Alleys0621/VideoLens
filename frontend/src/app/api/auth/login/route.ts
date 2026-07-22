import { NextRequest, NextResponse } from "next/server";
import {
  setAuthCookie,
  signToken,
  verifyPassword,
  USER_NAMESPACE,
  type PublicUser,
  type UserRecord,
} from "@/lib/auth";
import { getServerClient } from "@/lib/langgraph-client";

export const runtime = "nodejs";

/**
 * 密码登录.
 *
 * 注: 这里需要拿到 password_hash 做校验, 但 lib/auth.ts 出于安全只暴露 PublicUser.
 * 这一处特例直接调 Store 拿原始 record, 避免把 hash 暴露到通用 getUserByPhone API.
 */
async function findUserRecordByPhone(
  phone: string,
): Promise<{ id: string; rec: UserRecord } | null> {
  const client = getServerClient();
  const res = await client.store.searchItems([...USER_NAMESPACE], {
    filter: { phone },
    limit: 1,
  });
  const item = res.items?.[0];
  if (!item) return null;
  return { id: item.key, rec: item.value as UserRecord };
}

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
    const found = await findUserRecordByPhone(phone);
    // 统一报错避免手机号枚举攻击
    if (!found) {
      return NextResponse.json(
        { error: "手机号或密码错误" },
        { status: 401 },
      );
    }
    const ok = await verifyPassword(password, found.rec.password_hash);
    if (!ok) {
      return NextResponse.json(
        { error: "手机号或密码错误" },
        { status: 401 },
      );
    }
    const user: PublicUser = {
      id: found.id,
      phone: found.rec.phone,
      display_name: found.rec.display_name,
    };
    const token = await signToken({ sub: user.id, phone: user.phone });
    await setAuthCookie(token);
    return NextResponse.json({ user });
  } catch (e) {
    console.error("[auth/login] failed:", e);
    return NextResponse.json(
      { error: "登录失败, 请检查后端 LangGraph 服务是否正常" },
      { status: 503 },
    );
  }
}
