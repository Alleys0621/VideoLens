import { NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/auth";

export const runtime = "nodejs";

export async function GET() {
  const user = await getCurrentUser();
  // 未登录返回 200 + user:null, 而非 401.
  // 理由: /api/auth/me 是前端每次 mount 都会调用的「探活」接口,
  // 401 会在 DevTools 里刷一屏红字, 干扰真实错误排查。
  // 前端 AuthProvider.refresh() 已经按 user === null 判定未登录, 无需依赖 status code。
  if (!user) {
    return NextResponse.json({ user: null }, { status: 200 });
  }
  return NextResponse.json({ user });
}
