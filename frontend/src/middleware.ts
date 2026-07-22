import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { verifyToken, COOKIE_NAME } from "@/lib/jwt";

/**
 * 路由保护: 未登录用户跳 /login.
 *
 * middleware 跑在 edge runtime, 只引 edge-safe 的 lib/jwt.ts,
 * 不能引 lib/auth.ts (含 next/headers / bcryptjs / LangGraph SDK).
 */

const PUBLIC_PAGES = ["/login", "/register", "/forgot-password"];

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // 公开页面直接放行
  if (PUBLIC_PAGES.some((p) => pathname === p || pathname.startsWith(p + "/"))) {
    return NextResponse.next();
  }

  const token = req.cookies.get(COOKIE_NAME)?.value;
  if (!token) {
    return redirectToLogin(req);
  }

  const payload = await verifyToken(token);
  if (!payload) {
    return redirectToLogin(req);
  }

  return NextResponse.next();
}

function redirectToLogin(req: NextRequest) {
  const url = req.nextUrl.clone();
  const from = url.pathname + (url.search || "");
  url.pathname = "/login";
  url.search = `?from=${encodeURIComponent(from)}`;
  return NextResponse.redirect(url);
}

export const config = {
  /**
   * 匹配所有路径, 但排除:
   * - /api/*            后端 API (登录端点自己要能访问)
   * - /_next/static/*   Next.js 静态资源
   * - /_next/image/*    Next.js 图片优化
   * - /favicon.ico, 各种图片资源
   */
  matcher: [
    "/((?!api|_next/static|_next/image|favicon.ico|.*\\.(?:png|jpg|jpeg|svg|webp|ico|manifest)$).*)",
  ],
};
