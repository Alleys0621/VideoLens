import { SignJWT, jwtVerify } from "jose";

/**
 * JWT 工具 (edge-runtime safe).
 *
 * middleware 跑在 edge runtime, 不能引 next/headers / bcryptjs / LangGraph SDK
 * 这类 server-only / node-only 代码. 把纯 jose 的 JWT 逻辑独立在这里,
 * lib/auth.ts (server) 和 middleware.ts (edge) 都能引.
 */

export const COOKIE_NAME = "vl_token";
export const TOKEN_TTL_SEC = 7 * 24 * 60 * 60; // 7 天

const ENC = () => new TextEncoder();

function getSecret(): Uint8Array {
  const secret =
    process.env.JWT_SECRET ||
    "vl_dev_secret_change_me_in_production_please_0x9f2a";
  if (!process.env.JWT_SECRET && process.env.NODE_ENV === "production") {
    console.warn(
      "[auth] JWT_SECRET 未设置, 用了默认开发密钥. 生产环境必须在环境变量里设置 JWT_SECRET.",
    );
  }
  return ENC().encode(secret);
}

export interface TokenPayload {
  sub: string;
  phone: string;
}

export async function signToken(payload: TokenPayload): Promise<string> {
  return new SignJWT({ phone: payload.phone })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(payload.sub)
    .setIssuedAt()
    .setExpirationTime(`${TOKEN_TTL_SEC}s`)
    .sign(getSecret());
}

export async function verifyToken(token: string): Promise<TokenPayload | null> {
  try {
    const { payload } = await jwtVerify(token, getSecret(), {
      algorithms: ["HS256"],
    });
    if (typeof payload.sub !== "string" || typeof payload.phone !== "string") {
      return null;
    }
    return { sub: payload.sub, phone: payload.phone };
  } catch {
    return null;
  }
}
