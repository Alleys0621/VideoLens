import bcrypt from "bcryptjs";
import { cookies } from "next/headers";
import {
  COOKIE_NAME,
  TOKEN_TTL_SEC,
  signToken,
  verifyToken,
} from "./jwt";
import { query } from "./db";

/**
 * 认证工具 (server-side, 在 Next.js API Route / Server Component 内调).
 *
 * 注意: 本文件依赖 next/headers (cookies) + pg + bcryptjs,
 * 不能在 edge runtime (middleware) 直接引. middleware 只引 lib/jwt.ts.
 *
 * 存储后端: Postgres `users` 表 (db/init.sql 建表).
 *   id            UUID PK
 *   phone         VARCHAR(11) UNIQUE
 *   password_hash TEXT (bcrypt)
 *   display_name  VARCHAR(64)
 *   created_at    TIMESTAMPTZ
 *
 * 会话用 JWT (lib/jwt.ts) 存 HttpOnly cookie, 无状态.
 */

export { COOKIE_NAME, TOKEN_TTL_SEC, signToken, verifyToken };
export type { TokenPayload } from "./jwt";

export interface UserRecord {
  id: string;
  phone: string;
  password_hash: string;
  display_name: string;
  created_at: string;
}

export interface PublicUser {
  id: string;
  phone: string;
  display_name: string;
}

function toPublic(rec: UserRecord): PublicUser {
  return { id: rec.id, phone: rec.phone, display_name: rec.display_name };
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

/* ---------- Postgres CRUD (用户表) ---------- */

interface UserRow {
  id: string;
  phone: string;
  password_hash: string;
  display_name: string;
  created_at: string;
}

function rowToRecord(row: UserRow): UserRecord {
  return {
    id: row.id,
    phone: row.phone,
    password_hash: row.password_hash,
    display_name: row.display_name,
    // pg TIMESTAMPTZ 默认序列化为 ISO string
    created_at: typeof row.created_at === "string"
      ? row.created_at
      : new Date(row.created_at as unknown as Date).toISOString(),
  };
}

export async function getUserById(id: string): Promise<PublicUser | null> {
  try {
    const { rows } = await query<UserRow>(
      "SELECT id, phone, password_hash, display_name, created_at FROM users WHERE id = $1",
      [id],
    );
    if (rows.length === 0) return null;
    return toPublic(rowToRecord(rows[0]));
  } catch {
    return null;
  }
}

export async function getUserByPhone(
  phone: string,
): Promise<PublicUser | null> {
  const { rows } = await query<UserRow>(
    "SELECT id, phone, password_hash, display_name, created_at FROM users WHERE phone = $1",
    [phone],
  );
  if (rows.length === 0) return null;
  return toPublic(rowToRecord(rows[0]));
}

async function getUserRecordByPhone(
  phone: string,
): Promise<{ id: string; rec: UserRecord } | null> {
  const { rows } = await query<UserRow>(
    "SELECT id, phone, password_hash, display_name, created_at FROM users WHERE phone = $1",
    [phone],
  );
  if (rows.length === 0) return null;
  const rec = rowToRecord(rows[0]);
  return { id: rec.id, rec };
}

/**
 * 校验凭据 (phone + password) → 返回 PublicUser / null.
 * 专门给 /api/auth/login 用: 在内部完成密码校验, 不把 password_hash 暴露出去.
 */
export async function verifyCredentials(
  phone: string,
  password: string,
): Promise<PublicUser | null> {
  const found = await getUserRecordByPhone(phone);
  if (!found) return null;
  const ok = await verifyPassword(password, found.rec.password_hash);
  if (!ok) return null;
  return toPublic(found.rec);
}

async function getUserRecordById(
  id: string,
): Promise<{ id: string; rec: UserRecord } | null> {
  const { rows } = await query<UserRow>(
    "SELECT id, phone, password_hash, display_name, created_at FROM users WHERE id = $1",
    [id],
  );
  if (rows.length === 0) return null;
  const rec = rowToRecord(rows[0]);
  return { id: rec.id, rec };
}

export async function createUser(input: {
  phone: string;
  password: string;
  display_name?: string;
}): Promise<PublicUser> {
  const { randomUUID } = await import("crypto");
  const id = randomUUID();
  const passwordHash = await hashPassword(input.password);
  const displayName =
    input.display_name?.trim() || `用户_${input.phone.slice(-4)}`;
  const createdAt = new Date().toISOString();

  await query(
    `INSERT INTO users (id, phone, password_hash, display_name, created_at)
     VALUES ($1, $2, $3, $4, $5)`,
    [id, input.phone, passwordHash, displayName, createdAt],
  );

  return {
    id,
    phone: input.phone,
    display_name: displayName,
  };
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
  const newHash = await hashPassword(newPassword);
  await query(
    "UPDATE users SET password_hash = $1 WHERE id = $2",
    [newHash, userId],
  );
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
