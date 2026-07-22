import { NextResponse } from "next/server";

export const runtime = "nodejs";

/**
 * 忘记密码 (占位).
 *
 * 当前实现: 直接返回 501.
 * 短信通道开通后, 在这里补:
 *   1. body: { phone, code, new_password }
 *   2. 从 Store (["sms_codes"], key=phone) 读出验证码, 校验 body.code
 *   3. 通过则 hashPassword(new_password) → putItem 更新 user record
 *   4. 删除已用过的 sms_code
 *
 * 配合 /api/auth/sms (发送验证码) 一起工作.
 */
export async function POST() {
  return NextResponse.json(
    {
      error: "短信通道未开通, 无法通过验证码重置密码. 请联系管理员重置.",
      code: "SMS_NOT_AVAILABLE",
    },
    { status: 501 },
  );
}
