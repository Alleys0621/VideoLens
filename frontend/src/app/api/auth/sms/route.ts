import { NextResponse } from "next/server";

export const runtime = "nodejs";

/**
 * 发送短信验证码 (占位).
 *
 * 当前实现: 直接返回 501.
 * 接入阿里云短信后, 在这里补:
 *   1. body: { phone, scene: "login" | "register" | "reset" }
 *   2. 生成 6 位数字, 写入 Store (["sms_codes"], key=`${phone}:${scene}`, ttl=300s)
 *   3. 调阿里云 Dysmsapi (@alicloud/dysmsapi20170525) 发送
 *      模板: SMS_xxx (验证码模板), 签名: 已审批的签名
 *   4. 限流: 同 phone 60s 内不能重发 (Store ttl 自然支持)
 */
export async function POST() {
  return NextResponse.json(
    {
      error: "短信通道未开通, 验证码登录 / 找回密码暂不可用",
      code: "SMS_NOT_AVAILABLE",
    },
    { status: 501 },
  );
}
