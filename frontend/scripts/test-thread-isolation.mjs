/**
 * 多用户 thread 隔离测试.
 *
 * 用法: node frontend/scripts/test-thread-isolation.mjs
 *
 * 验证场景:
 *   用户 A 创建 thread (带 metadata.user_id=A.id)
 *   用户 B 创建 thread (带 metadata.user_id=B.id)
 *   - A 用 user_id=A 搜索 → 只看到 A 自己的 thread
 *   - B 用 user_id=B 搜索 → 只看到 B 自己的 thread
 *   - A 用 user_id=A 搜索 → 不应看到 B 的 thread
 *   - 不带 user_id 搜索 (旧式) → 看到所有 thread (兼容老前端, 但我们前端不再这样调)
 *
 * 测试通过 LangGraph 后端直接 API 调用 (绕过 Next.js, 因为前端 stream.submit
 * 是浏览器端 hook, 不易在脚本里模拟). 后端 API 行为 = 前端 SDK 行为.
 */

const BACKEND = process.env.BACKEND_URL || "http://localhost:2024";
const FRONTEND = process.env.FRONTEND_URL || "http://localhost:3000";

const ts = Date.now().toString().slice(-8);
const PHONE_A = `138${ts}`; // 11 位 (138 + 8 位时间戳)
const PHONE_B = `139${ts}`; // 11 位 (139 + 8 位时间戳)
const PASSWORD = "test123456";

let step = 0;
function section(name) {
  step += 1;
  console.log(`\n[${step}] ${name}`);
}
function assert(cond, msg) {
  if (cond) {
    console.log(`  ✓ ${msg}`);
    return true;
  }
  console.error(`  ✗ ${msg}`);
  process.exitCode = 1;
  return false;
}

async function callFe(path, opts = {}, cookie = "") {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (cookie) headers.Cookie = cookie;
  const res = await fetch(`${FRONTEND}${path}`, {
    method: opts.method || "GET",
    headers,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const setCookie = res.headers.get("set-cookie");
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  return { status: res.status, data, setCookie };
}

async function callBe(path, opts = {}) {
  const res = await fetch(`${BACKEND}${path}`, {
    method: opts.method || "GET",
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (res.status === 204) return { status: 204, data: null };
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  return { status: res.status, data };
}

async function login(phone) {
  const r = await callFe("/api/auth/login", {
    method: "POST",
    body: { phone, password: PASSWORD },
  });
  if (r.status !== 200) throw new Error(`login ${phone} 失败: ${JSON.stringify(r.data)}`);
  const m = /vl_token=[^;]+/.exec(r.setCookie || "");
  return { cookie: m ? m[0] : "", user: r.data.user };
}

async function createStoreUser(phone) {
  // 通过前端注册端点创建用户 (后端直接写 Store)
  await callFe("/api/auth/register", {
    method: "POST",
    body: { phone, password: PASSWORD, display_name: `测试-${phone.slice(-2)}` },
  });
}

async function createThread(userId, label) {
  // 模拟前端 stream.submit 时 SDK 自动调的 threads.create
  const r = await callBe("/threads", {
    method: "POST",
    body: {
      metadata: { user_id: userId },
    },
  });
  if (r.status !== 200) throw new Error(`createThread 失败: ${JSON.stringify(r.data)}`);
  // 写入一条测试消息, 模拟真实对话
  const threadId = r.data.thread_id;
  await callBe(`/threads/${threadId}/runs`, {
    method: "POST",
    body: {
      assistant_id: "agent",
      input: { messages: [{ role: "user", content: `测试消息: ${label}` }] },
      metadata: { user_id: userId },
      stream_mode: ["values"],
    },
  });
  return threadId;
}

async function searchThreads(userId) {
  // 模拟前端 getThreads 调用
  const r = await callBe("/threads/search", {
    method: "POST",
    body: {
      metadata: { user_id: userId },
      limit: 100,
    },
  });
  if (r.status !== 200) throw new Error(`searchThreads 失败: ${JSON.stringify(r.data)}`);
  return r.data;
}

async function deleteThread(threadId) {
  await callBe(`/threads/${threadId}`, { method: "DELETE" });
}

async function main() {
  console.log(`========================================`);
  console.log(` 多用户 Thread 隔离测试`);
  console.log(` 用户 A: ${PHONE_A}`);
  console.log(` 用户 B: ${PHONE_B}`);
  console.log(`========================================`);

  /* === 准备: 创建两个用户 === */
  section("创建测试用户 A + B");
  {
    await createStoreUser(PHONE_A);
    await createStoreUser(PHONE_B);
    const loginA = await login(PHONE_A);
    const loginB = await login(PHONE_B);
    assert(!!loginA.user?.id, `用户 A 登录成功 (id=${loginA.user?.id?.slice(0, 8)}...)`);
    assert(!!loginB.user?.id, `用户 B 登录成功 (id=${loginB.user?.id?.slice(0, 8)}...)`);

    /* === 创建 thread (带 user_id metadata) === */
    section("A 创建 thread, B 创建 thread");
    const threadA = await createThread(loginA.user.id, "from-A");
    const threadB = await createThread(loginB.user.id, "from-B");
    assert(!!threadA, `A 创建了 thread ${threadA.slice(0, 8)}...`);
    assert(!!threadB, `B 创建了 thread ${threadB.slice(0, 8)}...`);

    /* === A 视角搜索: 只看到自己的 === */
    section("A 视角 search (metadata.user_id=A)");
    {
      const list = await searchThreads(loginA.user.id);
      const ids = list.map((t) => t.thread_id);
      assert(ids.includes(threadA), `A 能看到自己的 thread ${threadA.slice(0, 8)}...`);
      assert(!ids.includes(threadB), `A 看不到 B 的 thread (隔离生效)`);
    }

    /* === B 视角搜索: 只看到自己的 === */
    section("B 视角 search (metadata.user_id=B)");
    {
      const list = await searchThreads(loginB.user.id);
      const ids = list.map((t) => t.thread_id);
      assert(ids.includes(threadB), `B 能看到自己的 thread ${threadB.slice(0, 8)}...`);
      assert(!ids.includes(threadA), `B 看不到 A 的 thread (隔离生效)`);
    }

    /* === 旧式搜索 (不带 user_id) === */
    section("旧式搜索 (不带 user_id, 兼容性)");
    {
      const r = await callBe("/threads/search", {
        method: "POST",
        body: { limit: 100 },
      });
      const ids = r.data.map((t) => t.thread_id);
      assert(ids.includes(threadA) && ids.includes(threadB),
        `旧式搜索能看到所有 thread (说明 thread 本身存在, 不是测试假阳性)`);
    }

    /* === 清理 === */
    section("清理测试数据");
    {
      await deleteThread(threadA);
      await deleteThread(threadB);
      // 删除两个测试用户
      for (const phone of [PHONE_A, PHONE_B]) {
        const sres = await callBe("/store/items/search", {
          method: "POST",
          body: {
            namespace_prefix: ["users"],
            filter: { phone },
            limit: 1,
          },
        });
        const item = sres.data.items?.[0];
        if (item) {
          await callBe("/store/items", {
            method: "DELETE",
            body: { namespace: ["users"], key: item.key },
          });
          console.log(`  ✓ 已清理用户 ${phone}`);
        }
      }
    }
  }

  console.log(`\n========================================`);
  console.log(process.exitCode ? ` ❌ 有失败用例` : ` ✅ 全部用例通过 — 多用户隔离生效`);
  console.log(`========================================\n`);
  process.exit(process.exitCode || 0);
}

main().catch((e) => {
  console.error("测试异常中断:", e);
  process.exit(1);
});
