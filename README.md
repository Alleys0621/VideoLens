# AlleysVid — AI 陪看智能体

> **V1.4.3** — Alleys 视角重建：工作记忆（video_time → 当前 event）+ 人设重构 + Mem0 质量优化
> 完整版本历史见 [VERSION.md](./VERSION.md)

边看剧边聊天的 AI 陪看搭子。选一集视频，Alleys 陪你一起看，随时聊剧情、问角色、吐槽笑点。

## 功能

- **AI 陪看聊天**：选集后直接对话，Alleys 基于剧情知识库回答
- **流式语音对话**：流式 ASR 实时转文字 + 流式 TTS 自动播报
- **用户系统**：手机号注册登录，每人独立的对话历史和记忆
- **工作记忆（V1.4.3）**：Alleys 根据播放位置感知当前事件，知道你看到哪、在讲什么，不假装看画面
- **播放进度记忆**：切走再回来从上次位置继续
- **观看状态上报**：前端定时上报视频坐标（V1.4.2）
- **分层用户画像**：L1 跨会话聊法偏好 + L2 单部作品角色偏好，第一视角注入
- **长期记忆**：Mem0 跨会话记住用户身份/偏好/立场（V1.4.3 重写提炼规则）
- **中文会话标题**：首条消息后自动起 4–12 字中文标题
- **人设**：三条铁律（情绪优先 / 认错 / 不编）+ 反机器人味，更像真人搭子
- **多设备访问**：本地或公网（cloudflared 隧道）
- **数据持久化**：Postgres 存用户/画像/进度/thread/对话 state，重启不丢

## 项目结构

```
VideoLens/
├── src/                       # 后端 (Python)
│   ├── agent/                 #   陪看智能体 (companion / 画像 / 记忆 / 语音)
│   ├── pipeline/              #   离线视频处理 (Stage1-3 → KB)
│   ├── server/                #   LangGraph graph + Postgres checkpointer
│   ├── app/                   #   Pipeline CLI 入口
│   ├── core/                  #   基础设施 (config / llm / logging / helpers)
│   ├── eval/                  #   质量评估
│   └── voiceprint/            #   声纹识别
├── frontend/                  # 前端 (Next.js)
│   └── src/{app, components, hooks, lib, providers}
├── config/                    # prompts.yaml + pipeline.yaml
├── db/                        # docker-compose + init.sql + migrations/
├── scripts/                   # 运维 + Pipeline 子阶段脚本
├── logs/                      # 所有 log (gitignore)
├── data/                      # 视频 + 流水线产物 (gitignore)
├── .tools/                    # cloudflared 二进制
├── start.bat                  # 一键启动
└── langgraph.json             # LangGraph dev 入口
```

各目录的 `__init__.py` docstring（后端）或 `README.md`（前端 / config / db / scripts）有更细的模块说明。

## Windows 部署

> 命令在 **PowerShell** 或 **CMD** 里逐条执行。所有步骤已在 Win11 + Docker Desktop 测试通过。

### 0. 系统依赖（首次部署才需要）

用 winget 一键装齐（管理员 PowerShell）：

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id OpenJS.NodeJS.LTS -e
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e
winget install --id Docker.DockerDesktop -e
```

- **Python 必须是 3.12.x**（不能 ≥3.13）。
- **Docker Desktop** 安装时会提示开启 WSL2，按默认下一步即可。装完 **手动启动一次 Docker Desktop**，等右下角/托盘图标变成 Running。
- 装完后 **重开一个终端**，让 PATH 生效。验证：
  ```powershell
  python --version ; node --version ; pnpm --version ; uv --version ; ffmpeg -version ; docker --version
  ```

pnpm 用 npm 装：
```powershell
npm install -g pnpm
```

### 1. 拉代码 + 装依赖

```powershell
# 任意目录
git clone https://github.com/Alleys0621/VideoLens.git
cd VideoLens          # 之后所有命令都在项目根 .\VideoLens 下
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
cd frontend
pnpm install
cd ..                 # 回到项目根
```

### 2. 配置环境变量

**必填**：阿里云百炼 API Key（系统环境变量，新开终端才生效）：
```powershell
# 任意目录都可以 (setx 写注册表)
setx DASHSCOPE_API_KEY "你的阿里云百炼APIKey"
```

**前端配置**（直接复制示例即可，`POSTGRES_URL` 默认值已对好本地 Docker）：
```powershell
# 项目根目录下
copy frontend\.env.local.example frontend\.env.local
```

可选：讯飞声纹（只在做 Pipeline 时才需要），在项目根目录 `.env` 里填 `XFYUN_APP_ID / XFYUN_API_KEY / XFYUN_API_SECRET`。

### 3. 放视频

把视频放到 `data/videos/{作品名}/{剧集名}.<ext>`，例如：
```
data/videos/家有儿女/第一季/第01集.mkv
```
仓库自带知识库 JSON，**不用跑 Pipeline**，但视频文件要自己放。

### 4. 启动 Postgres（Docker）

确保 Docker Desktop 已运行，然后：
```powershell
# 项目根目录下
docker compose -f db/docker-compose.yml up -d
```
- 容器名 `videolens-postgres`，映射 `127.0.0.1:25432`（避开本机 5432/15432 冲突）。
- 数据持久化在 Docker 卷 `videolens-pgdata`，`docker compose down` 不丢，只有 `down -v` 才删。

### 5. 配置 HTTPS（mkcert 本地证书，必做）

> 浏览器要求 HTTPS 才允许麦克风，语音对话必需。
> 协作者装一次根 CA 后，本机/局域网/公网都能用。**证书每次启动 `start.bat` 自动重签**，包含当前 WLAN IP，换 WiFi 重启即自适应。

**装 mkcert + 装本地根 CA**（一次性，项目根目录下）：

```powershell
# 项目根目录下
mkdir .tools -Force
Invoke-WebRequest "https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-windows-amd64.exe" -OutFile ".tools\mkcert.exe"

# 装本地根 CA (会弹 UAC, 点"是")
.tools\mkcert.exe -install
```

> 证书签发由 `start.bat` 调用 `scripts\renew-cert.ps1` 自动完成，不需要手动跑 mkcert。SAN 自动包含：`localhost` / `127.0.0.1` / `::1` / 电脑名 / `电脑名.local` / **当前 WLAN IP**。

### 6. （可选）安装 cloudflared 公网隧道

> 同 WiFi 用 `https://电脑名.local:3000` 或 `https://WLAN_IP:3000` 即可；只在需要公网访问时才装 cloudflared。
>
> 命令在**项目根**下执行。

从 GitHub Releases 自动下载最新 Windows 版到 `.tools/cloudflared.exe`：

```powershell
# 项目根目录下
.\scripts\install-cloudflared.bat
```

或手动下载（建议手动下载哈，因为要挂梯子的，不挂梯子下载巨慢）：访问 [cloudflare/cloudflared releases](https://github.com/cloudflare/cloudflared/releases)，下载 `cloudflared-windows-amd64.exe`，重命名为 `cloudflared.exe` 放到 `.tools/` 目录下。

### 7. 一键启动所有服务

```powershell
# 项目根目录下
.\start.bat
```

`start.bat` 会：
- `[0/7]` 调 `scripts\renew-cert.ps1` 重签证书（含当前 WLAN IP）
- `[1-7/7]` 启动 Postgres / LangGraph / Next.js(HTTPS) / ASR(wss) / TTS(wss) / 视频服务(https) / cloudflared

启动后浏览器访问：

| 入口 | URL |
|---|---|
| 笔记本本机 | `https://localhost:3000` |
| 同 WiFi iOS/Mac | `https://电脑名.local:3000`（mDNS 自动解析） |
| **同 WiFi Android** | `https://<笔记本 WLAN IP>:3000`（Android 不解析 `.local`，必须用 IP） |

> 笔记本 WLAN IP 查法：cmd 跑 `ipconfig`，找 "WLAN" 适配器的 IPv4，例如 `10.104.17.211`。
> 如果 `Alleys.local` 在 Android/iOS 解析到错 IP（虚拟网卡如 ZeroTier/WSL），改用 WLAN IP 直连即可。

> 停止：关掉弹出的各个 cmd 窗口；Postgres 用 `docker compose -f db/docker-compose.yml down`。

## 日常更新（每次 `git pull` 之后必做）

拉新代码后，依赖可能变了，**必须同步**，否则会莫名报错：

```powershell
# 项目根目录下
git pull
.venv\Scripts\activate
uv pip install -r requirements.txt
cd frontend
pnpm install
cd ..
```

如果 `db/docker-compose.yml` 也有改动，重建 Postgres 容器（数据不丢）：
```powershell
# 项目根目录下
docker compose -f db/docker-compose.yml up -d
```

**同步数据库 schema**（重要：每次 `git pull` 后都跑一次，幂等无害）：
```powershell
python -m scripts.apply_migrations
# 或先 dry-run 看看会跑哪些文件:
python -m scripts.apply_migrations --dry-run
```

> **新机器**不需要跑 —— `init.sql` 会被 docker-compose 首次启动时自动执行（表结构已含最新字段）。
> **已有库**每次 `git pull` 后跑 `python -m scripts.apply_migrations`，脚本按文件名顺序（0002→0003→…→最新）应用 `db/migrations/*.sql`。
> 所有 migration 文件都用 `IF NOT EXISTS` 幂等保护，重复执行无副作用，所以**每次 pull 后无脑跑一遍即可**，不用纠结"这次有没有 schema 变更"。

然后重新 `.\start.bat` 启动即可。

## 协作者访问（同局域网）

> 协作者 = 在自己设备上访问部署者跑的 AlleysVid。**每台设备装一次根 CA，永久有效**（直到换电脑或部署者重装 mkcert）。
>
> 前置：双方连同一个 WiFi。部署者告诉你他的电脑名（cmd 跑 `hostname`，例如 `Alleys`）。

### 1. 部署者分发根 CA

部署者从自己电脑拷贝根 CA 文件：
```
C:\Users\<部署者用户名>\AppData\Local\mkcert\rootCA.pem
```

**分发前必须重命名为带识别度的名字**（团队多人各自部署时避免混淆）：
```
alleysvid-ca-<姓名>.crt
```
例如 `alleysvid-ca-电脑名.crt`。

> ⚠️ 必须改扩展名为 `.crt`（不是 `.pem`）。Windows 双击 `.crt` 会直接打开证书安装向导；双击 `.pem` 会弹"用什么软件打开"。

通过 U 盘 / 微信 / 邮件发给协作者。

> 多个协作者各自部署 AlleysVid 时，每个人分发的根 CA 都不同。协作者电脑上可以并存多张 CA（按"主题 CN"区分，文件名不冲突即可）。

### 2. 装根 CA

> **重要**：装完 CA 后，**Android 用 WLAN IP 访问，iOS 用主机名 `.local` 访问**。原因：
> - iOS/macOS：原生 mDNS 解析 `.local`，证书含 `电脑名.local` ✓
> - **Android 不解析 `.local`**：必须用 `https://<部署者 WLAN IP>:3000`，证书每次 `start.bat` 启动都自动重签含当前 WLAN IP
>
> 部署者 WLAN IP 查法：cmd 跑 `ipconfig`，找 "WLAN" 适配器的 IPv4。

#### Windows

1. 双击收到的 `alleysvid-ca-xxx.crt` → "安装证书" → 选"本地计算机"（需管理员）→ 下一步
2. 选"将所有的证书都放入下列存储" → "浏览" → **受信任的根证书颁发机构** → 确定 → 下一步 → 完成
3. 重启 Chrome/Edge
4. 访问 `https://电脑名.local:3000`（例如 `https://Alleys.local:3000`），地址栏应显示🔒锁，无警告

> 如果双击 `.crt` 仍然弹"用什么软件打开"：右键 → 打开方式 → 选择"Cryptographic Open Extension"或 `certmgr.exe`；或直接 `Win+R` 跑 `certmgr.msc` → 操作 → 导入。

#### macOS

1. 双击 `alleysvid-ca-xxx.crt` → 自动打开"钥匙串访问"，添加到"登录"或"系统"钥匙串
2. 在钥匙串里搜 `mkcert` → 双击找到的证书 → 展开"信任" → "使用此证书时"改为**始终信任** → 关闭窗口（输密码确认）
3. 重启 Safari/Chrome
4. 访问 `https://电脑名.local:3000`

#### iOS

1. 把 `alleysvid-ca-xxx.crt` 通过邮件/AirDrop 发到手机，点击下载
2. **设置 → 通用 → VPN与设备管理** → 已下载的描述文件 → 点 install 安装
3. **设置 → 通用 → 关于本机 → 证书信任设置** → 把 mkcert 那一项的开关打开（**这步漏掉会导致 Safari 报"不是私密连接"且无法继续**）
4. Safari 访问 `https://电脑名.local:3000`
5. 注意：iOS Safari 的 MediaSource API 对音频支持差，TTS 流式播报可能不出声（视频/文字不受影响）

#### Android

1. 把 `alleysvid-ca-xxx.crt` 传到手机
2. **设置 → 安全 → 加密与凭据 → 安装证书 → CA 证书** → 选 `alleysvid-ca-xxx.crt`（会让输锁屏密码确认）
3. **Chrome 访问 `https://<部署者 WLAN IP>:3000`**（不要用 `.local`，Android 不解析）
   - 例如：`https://10.104.17.211:3000`
   - 部署者每次重启 `start.bat` 时如果 WLAN IP 变了，会自动重签证书含新 IP，协作者改用新 IP 即可

### 3. 验证

- 浏览器地址栏🔒锁，无"不安全"警告
- 按住麦克风按钮能录音（F12 console 应打 `[ASR] using AudioWorklet`）
- Alleys 回复能听到语音播报
- 视频流畅可 seek

### 4. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 地址栏红色"不安全" | CA 没装好 / 没装到"受信任的根证书颁发机构" | 重做步骤 2，确认选对存储；iOS 还要在"证书信任设置"里开开关 |
| `ERR_NAME_NOT_RESOLVED` (Android 用 .local) | Android 不解析 `.local` | 改用 `https://<WLAN IP>:3000` |
| `ERR_CONNECTION_REFUSED` (Android 用 .local) | mDNS 返回多个 IP（含 ZeroTier/WSL 虚拟网卡），Android 试到错的 | 同上，改用 WLAN IP |
| `ERR_CONNECTION_TIMED_OUT` | Windows 防火墙挡了 / 路由器 AP 隔离 | 防火墙放行 node.exe 给"专用 + 公用"；路由器后台关"AP 隔离" |
| `ERR_CONNECTION_REFUSED` (用 IP) | 服务没启动 / 笔记本 WLAN IP 变了 | 重启 `start.bat`；用最新 WLAN IP |
| `NET::ERR_CERT_*` (证书警告) | CA 没装 / 证书 SAN 不含访问的 IP | 装根 CA；部署者重启 start.bat 重签含新 IP |
| 麦克风按钮无反应 | 浏览器没授麦克风权限 | 地址栏🔒右侧 → 站点设置 → 麦克风 → 允许 |
| iOS 没声音 | MediaSource API 限制 | iOS 17.4+ 部分支持；老版本无解 |
| 换网络后访问不到 | 双方不在同一 WiFi | 切同一 WiFi；或部署者开 cloudflared 公网隧道 |

## 老数据迁移

如果你之前用过 `.langgraph_api/store.pckl`（V1.0 的内存存储），可一键把老用户和播放进度迁到 Postgres：

```bash
# 1. 确保 Postgres 已启动
# 2. 启动 LangGraph dev（:2024）
# 3. 运行迁移脚本
python -m scripts.migrate_to_postgres
```

迁移脚本会：
- 把 `store.pckl` 里的用户写入 `users` 表；
- 把播放进度写入 `playback_progress` 表；
- 把有效的 thread 元数据同步到 `threads` 表；
- 列出无 user_id 或空对话的孤儿 thread。

清理孤儿 thread：
```bash
echo y | python -m scripts.cleanup_orphan_threads
```

## 技术栈

- **AI 对话**：LangGraph + Qwen（通义千问）
- **主回复 LLM**：qwen3.7-flash（默认，快）/ qwen3.7-plus（`COMPANION_MAIN_MODEL=plus` 切高质量）
- **画像 / 会话标题**：qwen3.7-flash / qwen-turbo
- **Embedding**：qwen3.7-text-embedding（检索向量 + Mem0）
- **长期记忆**：Mem0（Qdrant 本地向量库）
- **语音识别**：DashScope paraformer-realtime-v2（流式 WebSocket）
- **语音合成**：DashScope qwen-audio-3.0-tts-flash + longanhuan_v3.6（流式 WebSocket）
- **前端**：Next.js + TailwindCSS + Artplayer
- **数据存储**：Postgres（用户 / 画像 / 播放进度 / watching_state / thread 元数据 / LangGraph checkpoints）
- **Checkpointer**：`langgraph-checkpoint-postgres`（`AsyncPostgresSaver`）
- **公网隧道**：cloudflared quick tunnel

## 数据布局

```
data/
├── videos/{作品名}/{剧集}          # 输入视频
├── output/{video_dir}/             # 流水线产物 (audio.json / visual.json / stage3_kb.json ...)
└── output/_global/                 # 跨集全局产物 (characters / global_arcs / character_profiles)
```

Postgres 中的业务表：
- `users`：用户账号
- `user_profiles`：L1 跨会话用户画像（聊法偏好、剧透接受度、接梗浓度、engagement_motivation、alleys_attitude）
- `show_profiles`：L2 作品级画像（喜欢的角色、角色评价、主题偏好）
- `playback_progress`：用户 × 视频的播放进度
- `watching_state`：用户实时的视频坐标（video_time / is_playing）
- `threads`：会话元数据（`custom_title` 为中文自动标题）
- `checkpoints` / `checkpoint_writes` / `checkpoint_blobs`：由 `AsyncPostgresSaver.setup()` 自动创建，存储对话 state

仓库自带知识库 JSON（`data/output/*/stage3_kb.json`）、全局角色 / 剧情弧（`data/output/_global/`）、声纹 / OCR / 场景数据，clone 后开箱即用。**不含**视频文件，需自行放入 `data/videos/`。

## License

MIT
