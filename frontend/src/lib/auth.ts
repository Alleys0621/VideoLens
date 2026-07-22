import bcrypt from "bcryptjs";
import { cookies } from "next/headers";
import { getServerClient } from "./langgraph-client";
import {
  COOKIE_NAME,
  TOKEN_TTL_SEC,
  signToken,
  verifyToken,
} from "./jwt";

/**
 * 认证工具 (server-side, 在 Next.js API Route / Server Component 内调).
 *
 * 注意: 本文件依赖 next/headers (cookies) + LangGraph SDK + bcryptjs,
 * 不能在 edge runtime (middleware) 直接引. middleware 只引 lib/jwt.ts.
 *
 * 存储复用 LangGraph Store (namespace KV):
 *   namespace = ["users"]
 *   key       = user_id (uuid v4)
 *   value     = UserRecord
 *
 * 会话用 JWT (lib/jwt.ts) 存 HttpOnly cookie, 无状态.
 */

export { COOKIE_NAME, TOKEN_TTL_SEC, signToken, verifyToken };
export type { TokenPayload } from "./jwt";

export const USER_NAMESPACE = ["users"] as const;

export interface UserRecord {
  phone: string;
  password_hash: string;
  display_name: string;
  created_at: string;
  // 后续扩展: wx_openid?, avatar_url?, ...
  // 索引签名: 让 putItem 接受 (LangGraph SDK 期望 Record<string, unknown>)
  [key: string]: unknown;
}

export interface PublicUser {
  id: string;
  phone: string;
  display_name: string;
}

function toPublic(id: string, rec: UserRecord): PublicUser {
  return { id, phone: rec.phone, display_name: rec.display_name };
}

/* ---------- 密码 ---------- */

const SALT_ROUNDS = 10;

export async function hashPassword(plain: string): Promise<string> {
  return bcrypt.hash(plain, SALT_ROUNDS);
}

export async function verifyPassword(
  plain: string,
  hash: string,
): Promise<boolean> {
  return bcrypt.compare(plain, hash);
}

/* ---------- Store CRUD (用户表) ---------- */

export async function getUserById(id: string): Promise<PublicUser | null> {
  const client = getServerClient();
  try {
    const item = await client.store.getItem([...USER_NAMESPACE], id);
    if (!item) return null;
    return toPublic(id, item.value as UserRecord);
  } catch {
    return null;
  }
}

export async function getUserByPhone(
  phone: string,
): Promise<PublicUser | null> {
  const client = getServerClient();
  const res = await client.store.searchItems([...USER_NAMESPACE], {
    filter: { phone },
    limit: 1,
  });
  const item = res.items?.[0];
  if (!item) return null;
  return toPublic(item.key, item.value as UserRecord);
}

async function getUserRecordByPhone(
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

async function getUserRecordById(
  id: string,
): Promise<{ id: string; rec: UserRecord } | null> {
  const client = getServerClient();
  const item = await client.store.getItem([...USER_NAMESPACE], id);
  if (!item) return null;
  return { id, rec: item.value as UserRecord };
}

export async function createUser(input: {
  phone: string;
  password: string;
  display_name?: string;
}): Promise<PublicUser> {
  const client = getServerClient();
  const { randomUUID } = await import("crypto");
  const id = randomUUID();
  const rec: UserRecord = {
    phone: input.phone,
    password_hash: await hashPassword(input.password),
    display_name:
      input.display_name?.trim() || `用户_${input.phone.slice(-4)}`,
    created_at: new Date().toISOString(),
  };
  await client.store.putItem([...USER_NAMESPACE], id, rec);
  return toPublic(id, rec);
}

/** 修改密码: 返回 true=成功, false=旧密码错误 / 用户不存在 */
export async function changePassword(
  userId: string,
  oldPassword: string,
  newPassword: string,
): Promise<boolean> {
  const existing = await getUserRecordById(userId);
  if (!existing) return false;
  const ok = await verifyPassword(oldPassword, existing.rec.password_hash);
  if (!ok) return false;
  const updated: UserRecord = {
    ...existing.rec,
    password_hash: await hashPassword(newPassword),
  };
  const client = getServerClient();
  await client.store.putItem([...USER_NAMESPACE], userId, updated);
  return true;
}

/* ---------- Cookie (HttpOnly) ---------- */

export async function setAuthCookie(token: string): Promise<void> {
  const store = await cookies();
  store.set(COOKIE_NAME, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: TOKEN_TTL_SEC,
  });
}

export async function clearAuthCookie(): Promise<void> {
  const store = await cookies();
  store.delete(COOKIE_NAME);
}

export async function getCurrentUser(): Promise<PublicUser | null> {
  const store = await cookies();
  const token = store.get(COOKIE_NAME)?.value;
  if (!token) return null;
  const payload = await verifyToken(token);
  if (!payload) return null;
  return getUserById(payload.sub);
}
