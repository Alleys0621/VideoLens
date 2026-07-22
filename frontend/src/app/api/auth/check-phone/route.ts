import { NextRequest, NextResponse } from "next/server";
import { getUserByPhone } from "@/lib/auth";

export const runtime = "nodejs";

const PHONE_RE = /^1[3-9]\d{9}$/;

/**
 * 检查手机号是否已注册.
 *
 * 用于登录页/注册页的失焦提示:
 *   - 登录页: 未注册时提示"该手机号未注册, 去注册"
 *   - 注册页: 已注册时提示"已注册, 去登录"
 *
 * 注意: 这会引入轻度手机号枚举风险 (任何人都可探哪些号注册过),
 * 但 UX 上 "未注册提前提示" 的取舍是合理的, 企业级应用也这么做.
 */
export async function GET(req: NextRequest) {
  const phone = req.nextUrl.searchParams.get("phone")?.trim() || "";
  if (!PHONE_RE.test(phone)) {
    return NextResponse.json(
      { error: "手机号格式不正确" },
      { status: 400 },
    );
  }
  try {
    const user = await getUserByPhone(phone);
    return NextResponse.json({ exists: !!user });
  } catch (e) {
    console.error("[auth/check-phone] failed:", e);
    return NextResponse.json(
      { error: "检查失败, 请稍后重试" },
      { status: 503 },
    );
  }
}
