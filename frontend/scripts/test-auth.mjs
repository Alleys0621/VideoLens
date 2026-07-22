/**
 * 认证系统端到端测试.
 *
 * 用法:
 *   node frontend/scripts/test-auth.mjs
 *
 * 前置条件:
 *   1. 后端 langgraph dev 跑在 :2024
 *   2. 前端 next dev 跑在 :3000
 *   3. frontend/.env.local 配了 LANGGRAPH_API_URL
 *
 * 测试覆盖:
 *   - check-phone (未注册/已注册/格式错)
 *   - register (新号/重复/密码短/格式错)
 *   - login (不存在/密码错/正确/空字段)
 *   - me (带 cookie / 不带 cookie)
 *   - change-password (旧密码错/新密码短/正确)
 *   - login 新旧密码 (旧失败/新成功)
 *   - logout + logout 后 me
 *
 * 测试间用唯一手机号隔离, 不污染真实用户数据.
 */

const BASE = process.env.BASE_URL || "http://localhost:3000";

// 用时间戳生成合法手机号 (138 + 8 位) — 跑多次也不冲突
const ts = Date.now().toString().slice(-8);
const TEST_PHONE = `138${ts}`;
const TEST_PASSWORD = "test123456";
const TEST_NEW_PASSWORD = "newpass654321";
const TEST_NAME = "E2E测试用户";

let cookie = "";

/* ---------- 工具 ---------- */

async function call(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (cookie) headers.Cookie = cookie;
  const res = await fetch(`${BASE}${path}`, {
    method: opts.method || "GET",
    headers,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  // 捕获 set-cookie
  const setCookie = res.headers.get("set-cookie");
  if (setCookie) {
    const m = /vl_token=[^;]+/.exec(setCookie);
    if (m) cookie = m[0];
  }
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  return { status: res.status, data, headers: res.headers };
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

let step = 0;
function section(name) {
  step += 1;
  console.log(`\n[${step}] ${name}`);
}

/* ---------- 测试用例 ---------- */

async function main() {
  console.log(`========================================`);
  console.log(` 认证 E2E 测试 — ${TEST_PHONE}`);
  console.log(` 后端: ${BASE}`);
  console.log(`========================================`);

  /* === check-phone: 未注册 === */
  section("check-phone 未注册号");
  {
    const r = await call(`/api/auth/check-phone?phone=${TEST_PHONE}`);
    assert(r.status === 200, `status 200 (实际 ${r.status})`);
    assert(r.data?.exists === false, `exists=false`);
  }

  /* === check-phone: 格式错 === */
  section("check-phone 格式错");
  {
    const r = await call(`/api/auth/check-phone?phone=12345`);
    assert(r.status === 400, `status 400 (实际 ${r.status})`);
  }

  /* === register: 新号 === */
  section("register 新号");
  {
    const r = await call("/api/auth/register", {
      method: "POST",
      body: {
        phone: TEST_PHONE,
        password: TEST_PASSWORD,
        display_name: TEST_NAME,
      },
    });
    assert(r.status === 200, `status 200 (实际 ${r.status}, ${JSON.stringify(r.data)})`);
    assert(!!r.data?.user?.id, `返回 user.id`);
    assert(r.data?.user?.phone === TEST_PHONE, `phone 匹配`);
    assert(r.data?.user?.display_name === TEST_NAME, `display_name 匹配`);
    // 注册后不应自动登录 (无 cookie)
    assert(!cookie, `注册后不应自动设置 cookie (实际 cookie=${cookie || "无"})`);
  }

  /* === register: 重复 === */
  section("register 重复号");
  {
    const r = await call("/api/auth/register", {
      method: "POST",
      body: { phone: TEST_PHONE, password: TEST_PASSWORD },
    });
    assert(r.status === 409, `status 409 (实际 ${r.status})`);
    assert(/已注册/.test(r.data?.error || ""), `错误信息含"已注册"`);
  }

  /* === register: 密码短 === */
  section("register 密码短");
  {
    // 用合法但未注册的号 (137开头), 触发密码长度校验而不是手机号格式校验
    const r = await call("/api/auth/register", {
      method: "POST",
      body: { phone: `137${ts}`, password: "123" },
    });
    assert(r.status === 400, `status 400 (实际 ${r.status})`);
    assert(/密码至少 6 位/.test(r.data?.error || ""), `错误信息含"密码至少 6 位"`);
  }

  /* === register: 手机号格式错 === */
  section("register 手机号格式错");
  {
    const r = await call("/api/auth/register", {
      method: "POST",
      body: { phone: "abc12345", password: TEST_PASSWORD },
    });
    assert(r.status === 400, `status 400 (实际 ${r.status})`);
    assert(/手机号格式/.test(r.data?.error || ""), `错误信息含"手机号格式"`);
  }

  /* === check-phone: 已注册 === */
  section("check-phone 已注册");
  {
    const r = await call(`/api/auth/check-phone?phone=${TEST_PHONE}`);
    assert(r.status === 200, `status 200 (实际 ${r.status})`);
    assert(r.data?.exists === true, `exists=true`);
  }

  /* === me: 未登录 === */
  section("me 未登录");
  {
    cookie = ""; // 清空 cookie
    const r = await call("/api/auth/me");
    assert(r.status === 401, `status 401 (实际 ${r.status})`);
    assert(r.data?.user === null, `user=null`);
  }

  /* === login: 不存在的号 === */
  section("login 不存在的手机号");
  {
    const r = await call("/api/auth/login", {
      method: "POST",
      body: { phone: "13999999999", password: "anything" },
    });
    assert(r.status === 401, `status 401 (实际 ${r.status})`);
    assert(/手机号或密码错误/.test(r.data?.error || ""), `统一报错 (避免枚举)`);
    assert(!cookie, `未设 cookie`);
  }

  /* === login: 密码错 === */
  section("login 密码错误");
  {
    const r = await call("/api/auth/login", {
      method: "POST",
      body: { phone: TEST_PHONE, password: "wrong_password" },
    });
    assert(r.status === 401, `status 401 (实际 ${r.status})`);
    assert(/手机号或密码错误/.test(r.data?.error || ""), `统一报错`);
  }

  /* === login: 空字段 === */
  section("login 空字段");
  {
    const r = await call("/api/auth/login", {
      method: "POST",
      body: { phone: "", password: "" },
    });
    assert(r.status === 400, `status 400 (实际 ${r.status})`);
  }

  /* === login: 正确 === */
  section("login 正确密码");
  {
    const r = await call("/api/auth/login", {
      method: "POST",
      body: { phone: TEST_PHONE, password: TEST_PASSWORD },
    });
    assert(r.status === 200, `status 200 (实际 ${r.status}, ${JSON.stringify(r.data)})`);
    assert(!!r.data?.user?.id, `返回 user.id`);
    assert(/vl_token=[^;]+/.test(cookie || ""), `已设置 cookie`);
  }

  /* === me: 已登录 === */
  section("me 已登录");
  {
    const r = await call("/api/auth/me");
    assert(r.status === 200, `status 200 (实际 ${r.status})`);
    assert(r.data?.user?.phone === TEST_PHONE, `phone 匹配`);
    assert(r.data?.user?.display_name === TEST_NAME, `display_name 匹配`);
  }

  /* === change-password: 旧密码错 === */
  section("change-password 旧密码错误");
  {
    const r = await call("/api/auth/change-password", {
      method: "POST",
      body: { old_password: "wrong", new_password: TEST_NEW_PASSWORD },
    });
    assert(r.status === 401, `status 401 (实际 ${r.status})`);
    assert(/旧密码错误/.test(r.data?.error || ""), `错误信息含"旧密码错误"`);
  }

  /* === change-password: 新密码太短 === */
  section("change-password 新密码太短");
  {
    const r = await call("/api/auth/change-password", {
      method: "POST",
      body: { old_password: TEST_PASSWORD, new_password: "123" },
    });
    assert(r.status === 400, `status 400 (实际 ${r.status})`);
    assert(/密码至少 6 位/.test(r.data?.error || ""), `错误信息含"密码至少 6 位"`);
  }

  /* === change-password: 正确 === */
  section("change-password 正确");
  {
    const r = await call("/api/auth/change-password", {
      method: "POST",
      body: {
        old_password: TEST_PASSWORD,
        new_password: TEST_NEW_PASSWORD,
      },
    });
    assert(r.status === 200, `status 200 (实际 ${r.status}, ${JSON.stringify(r.data)})`);
    assert(r.data?.ok === true, `ok=true`);
  }

  /* === login: 用旧密码应失败 === */
  section("login 旧密码已失效");
  {
    const r = await call("/api/auth/login", {
      method: "POST",
      body: { phone: TEST_PHONE, password: TEST_PASSWORD },
    });
    assert(r.status === 401, `status 401 (实际 ${r.status})`);
  }

  /* === login: 用新密码 === */
  section("login 新密码");
  {
    const r = await call("/api/auth/login", {
      method: "POST",
      body: { phone: TEST_PHONE, password: TEST_NEW_PASSWORD },
    });
    assert(r.status === 200, `status 200 (实际 ${r.status})`);
    assert(/vl_token=[^;]+/.test(cookie || ""), `已设置新 cookie`);
  }

  /* === logout === */
  section("logout");
  {
    const r = await call("/api/auth/logout", { method: "POST" });
    assert(r.status === 200, `status 200 (实际 ${r.status})`);
    // logout 应清除 cookie (Set-Cookie: vl_token=; Max-Age=0)
    const sc = r.headers.get("set-cookie") || "";
    assert(/vl_token=;|vl_token=,|Max-Age=0/.test(sc), `set-cookie 清除 token (实际 ${sc.slice(0, 100)})`);
    cookie = "";
  }

  /* === me: 已登出 === */
  section("me 已登出");
  {
    const r = await call("/api/auth/me");
    assert(r.status === 401, `status 401 (实际 ${r.status})`);
    assert(r.data?.user === null, `user=null`);
  }

  /* === 占位端点 (sms / forgot-password / wechat) === */
  section("占位端点返回 501");
  {
    const sms = await call("/api/auth/sms", {
      method: "POST",
      body: { phone: TEST_PHONE, scene: "reset" },
    });
    assert(sms.status === 501, `sms status 501 (实际 ${sms.status})`);

    const forgot = await call("/api/auth/forgot-password", {
      method: "POST",
      body: { phone: TEST_PHONE, code: "123456", new_password: "x" },
    });
    assert(forgot.status === 501, `forgot-password status 501 (实际 ${forgot.status})`);

    const wx = await call("/api/auth/wechat/callback");
    assert(wx.status === 501, `wechat/callback status 501 (实际 ${wx.status})`);
  }

  /* === 清理测试用户 === */
  section("清理测试数据");
  {
    // 直接调后端 Store 删除测试用户, 避免污染
    try {
      const searchRes = await fetch(`${BASE.replace(":3000", ":2024")}/store/items/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          namespace_prefix: ["users"],
          filter: { phone: TEST_PHONE },
          limit: 1,
        }),
      });
      const searchData = await searchRes.json();
      const item = searchData.items?.[0];
      if (item) {
        await fetch(`${BASE.replace(":3000", ":2024")}/store/items`, {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ namespace: ["users"], key: item.key }),
        });
        console.log(`  ✓ 已清理测试用户 (key=${item.key})`);
      } else {
        console.log(`  (无测试数据需清理)`);
      }
    } catch (e) {
      console.log(`  (清理跳过: ${e.message})`);
    }
  }

  console.log(`\n========================================`);
  console.log(process.exitCode ? ` ❌ 有失败用例` : ` ✅ 全部用例通过`);
  console.log(`========================================\n`);
  process.exit(process.exitCode || 0);
}

main().catch((e) => {
  console.error("测试异常中断:", e);
  process.exit(1);
});
