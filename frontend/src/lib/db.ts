import { Pool, type PoolClient, type QueryResult, type QueryResultRow } from "pg";

/**
 * Postgres 连接池单例 (server-side only, 在 Next.js API Route / Server Component 内调).
 *
 * 不能在 edge runtime 用 (依赖 pg native). 路由文件请加 `export const runtime = "nodejs"`.
 *
 * 连接串从环境变量读:
 *   POSTGRES_URL  — 必填, 如 postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens
 *   DATABASE_URI  — 备用别名 (与 LangGraph 共用)
 */

let _pool: Pool | null = null;

function getConnString(): string {
  const url =
    process.env.POSTGRES_URL ||
    process.env.DATABASE_URI ||
    // 默认值与 db/docker-compose.yml 一致, 仅本地开发兜底
    "postgresql://videolens:videolens_dev@127.0.0.1:25432/videolens";
  return url;
}

export function getPool(): Pool {
  if (_pool) return _pool;
  _pool = new Pool({
    connectionString: getConnString(),
    // dev 模式连接池小一点够用; 生产再调
    max: 10,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 5000,
  });
  // 不让单个查询错误搞崩整个进程
  _pool.on("error", (err) => {
    console.error("[pg pool] idle client error", err);
  });
  return _pool;
}

/**
 * 跑一个参数化查询, 返回 result.
 * 用法: const { rows } = await query<UserRow>('SELECT * FROM users WHERE id = $1', [id]);
 */
export async function query<T extends QueryResultRow = QueryResultRow>(
  text: string,
  params: unknown[] = [],
): Promise<QueryResult<T>> {
  const pool = getPool();
  return pool.query<T>(text, params as never);
}

/**
 * 跑一个事务回调. 回调内拿 client, 任一 await 抛错自动 ROLLBACK.
 * 用法:
 *   await withTransaction(async (client) => {
 *     await client.query('INSERT ...');
 *     await client.query('UPDATE ...');
 *   });
 */
export async function withTransaction<T>(
  fn: (client: PoolClient) => Promise<T>,
): Promise<T> {
  const pool = getPool();
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const result = await fn(client);
    await client.query("COMMIT");
    return result;
  } catch (e) {
    await client.query("ROLLBACK").catch(() => {});
    throw e;
  } finally {
    client.release();
  }
}
