import { NextResponse } from "next/server";

export const runtime = "nodejs";

/**
 * 微信扫码登录 OAuth 回调 (占位).
 *
 * 当前实现: 直接返回 501.
 * 接入微信开放平台网站应用后, 在这里补:
 *   1. 拿到 ?code=xxx (微信扫码后跳回此 URL 携带 code)
 *   2. POST https://api.weixin.qq.com/sns/oauth2/access_token
 *      ?appid=APP_ID&secret=APP_SECRET&code=xxx&grant_type=authorization_code
 *      → 拿到 access_token + openid + unionid
 *   3. 用 openid 在 Store 查 user (["users"], filter: { wx_openid })
 *      不存在则自动创建 (无密码, 无手机号, 仅 openid)
 *   4. 签 JWT, setAuthCookie, 重定向回前端首页
 *
 * 需要环境变量: WECHAT_APP_ID / WECHAT_APP_SECRET / WECHAT_REDIRECT_URI
 */
export async function GET() {
  return NextResponse.json(
    {
      error: "微信扫码登录未开通, 请使用手机号密码登录",
      code: "WECHAT_NOT_AVAILABLE",
    },
    { status: 501 },
  );
}
