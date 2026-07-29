# AlleysVid — AI 陪看智能体

> **V1.2.1** — 路由与模型层优化：意图路由 hybrid 化、主 LLM 按置信度切 flash/plus、embedding 升级、配置统一进 yaml

边看剧边聊天的 AI 陪看搭子。选一集视频，Alleys 陪你一起看，随时聊剧情、问角色、吐槽笑点。

## 版本记录

### V1.2.1（2026-07-29）

**意图路由 hybrid 化**
- 新增 `hybrid_route_intent`：embedding 优先，cosine < 阈值时回退 LLM
- 实测 50 条混合数据：准确率 96%（vs 纯 LLM 87.5%、纯 emb 84%），70% query 走 emb 直通，平均 485ms（vs 纯 LLM 1976ms，快 4 倍）
- 默认阈值 0.65（基于数据扫描甜点位）

**主 LLM 按置信度切 flash/plus**
- intent 置信度 ≥ 0.75 → qwen3.7-flash（首 token 快 2-4 倍）
- 否则 → qwen3.7-plus（防幻觉）
- 26 条 task×cosine 矩阵测试定位 flash 崩塌边界：低置信区间 flash 易幻觉/越界，plus 更稳

**Embedding 升级**
- `text-embedding-v3` → `qwen3.7-text-embedding`
- 100 条数据标定：两模型 cosine 分布几乎一致（中位漂移 0.010），所有阈值上 qwen3.7 准确率都 ≥ v3
- 阈值不需要重新校准

**配置统一进 yaml**
- 路由/模型相关配置全部从 `.env` 迁到 `config/pipeline.yaml`
- `.env` 只留 `DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL`

### V1.2（2026-06）

陪看智能体心智层升级：语义意图理解、上下文白名单、分层用户画像、中文会话标题。

## 功能

- **AI 陪看聊天**：选集后直接对话，Alleys 基于剧情知识库回答你的问题
- **流式语音对话**：说话给 Alleys 听（流式 ASR 实时转文字），Alleys 回答自动语音播报（流式 TTS）
- **用户系统**：手机号注册登录，每人独立的对话历史和记忆
- **播放进度记忆**：切走再回来，从上次的位置继续看
- **语义意图理解**：qwen-max 同时识别用户任务类型、当下情绪和一句话状态，让回复更对症
- **Hybrid 意图路由（V1.2.1）**：embedding 优先 + LLM 兜底，准确率 96%，70% query 直通 emb（~1ms），整体路由快 4 倍
- **主 LLM 按置信度切换（V1.2.1）**：高置信走 qwen3.7-flash（快），低置信走 qwen3.7-plus（稳）
- **上下文白名单（Context Budget）**：按任务类型决定 Alleys 本轮能看到哪些剧情上下文，避免闲聊时硬塞知识库
- **分层用户画像**：L1 跨会话聊法偏好 + L2 单部作品角色偏好，记忆跟着人走，但不喧宾夺主
- **中文会话标题**：首条消息后自动起一个 4–12 字中文标题，方便历史会话列表辨认
- **更自然的人设**：INFJ/ENFP 混合人格 + 反机器人味 + 反幻觉约束，Alleys 更像真人搭子
- **多设备访问**：本地或公网都能用（cloudflared 隧道）
- **数据持久化**：Postgres 存储用户、画像、播放进度、thread 元数据及对话 state，LangGraph 重启不丢

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
git clone https://github.com/Alleys0621/VideoLens.git
cd VideoLens
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
cd frontend
pnpm install
cd ..
```

### 2. 配置环境变量

**必填**：阿里云百炼 API Key（系统环境变量，新开终端才生效）：
```powershell
setx DASHSCOPE_API_KEY "你的阿里云百炼APIKey"
```

**前端配置**（直接复制示例即可，`POSTGRES_URL` 默认值已对好本地 Docker）：
```powershell
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
docker compose -f db/docker-compose.yml up -d
```
- 容器名 `videolens-postgres`，映射 `127.0.0.1:25432`（避开本机 5432/15432 冲突）。
- 数据持久化在 Docker 卷 `videolens-pgdata`，`docker compose down` 不丢，只有 `down -v` 才删。

### 5. （可选）安装 cloudflared 公网隧道

> 只在需要**公网访问**时才需要；同一 WiFi/内网直接用 `http://localhost:3000` 或局域网 IP 即可。

从 GitHub Releases 自动下载最新 Windows 版到 `.tools/cloudflared.exe`：

```powershell
.\scripts\install-cloudflared.bat
```

或手动下载：访问 [cloudflare/cloudflared releases](https://github.com/cloudflare/cloudflared/releases)，下载 `cloudflared-windows-amd64.exe`，重命名为 `cloudflared.exe` 放到 `.tools/` 目录下。

### 6. 一键启动所有服务

```powershell
.\start.bat
```

`start.bat` 会弹出多个窗口分别启动：
- Backend LangGraph `:2024`
- Frontend Next.js `:3000`
- ASR `:9800`
- TTS `:9801`
- cloudflared 隧道（第 5 步已装则自动启动；不要公网可忽略）

浏览器打开 http://localhost:3000 → 注册登录 → 选一集 → 开聊。

> 停止：关掉弹出的各个 cmd 窗口；Postgres 用 `docker compose -f db/docker-compose.yml down`。

### 日常更新（每次 `git pull` 之后必做）

拉新代码后，依赖可能变了，**必须同步**，否则会莫名报错：

```powershell
git pull
.venv\Scripts\activate
uv pip install -r requirements.txt
cd frontend
pnpm install
cd ..
```

如果 `db/docker-compose.yml` 也有改动，重建 Postgres 容器（数据不丢）：
```powershell
docker compose -f db/docker-compose.yml up -d
```

然后重新 `.\start.bat` 启动即可。

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
- **意图路由**：qwen3.7-flash（LLM 兜底）+ embedding 余弦匹配（直通），hybrid 模式
- **主回复 LLM**：qwen3.7-plus（默认/低置信）+ qwen3.7-flash（高置信），按 intent 置信度切换
- **Embedding**：qwen3.7-text-embedding（意图路由 catalogue + 检索向量）
- **语义理解**：qwen-max（任务 / 情绪 / 状态）
- **语音识别**：DashScope paraformer-realtime-v2（流式 WebSocket）
- **语音合成**：DashScope qwen-audio-3.0-tts-flash + longanhuan_v3.6（流式 WebSocket）
- **前端**：Next.js + TailwindCSS + Artplayer
- **数据存储**：Postgres（用户、画像、播放进度、thread 元数据、LangGraph checkpoints）
- **Checkpointer**：`langgraph-checkpoint-postgres`（`AsyncPostgresSaver`）
- **公网隧道**：cloudflared quick tunnel

## 数据布局

```
data/
├── videos/{作品名}/{剧集}          # 输入视频
├── output/{video_dir}/             # 流水线产物 (audio.json / visual.json / stage3_dryrun.json ...)
└── output/_global/                 # 跨集全局产物 (characters / global_arcs / character_profiles)
```

Postgres 中的业务表：
- `users`：用户账号
- `user_profiles`：L1 跨会话用户画像（聊法偏好、剧透接受度、接梗浓度）
- `show_profiles`：L2 作品级画像（喜欢的角色、角色评价）
- `playback_progress`：用户 × 视频的播放进度
- `threads`：会话元数据（`custom_title` 为中文自动标题）
- `checkpoints` / `checkpoint_writes` / `checkpoint_blobs`：由 `AsyncPostgresSaver.setup()` 自动创建，存储对话 state

## 包含的数据

仓库自带以下数据，clone 后开箱即用（无需重新跑 Pipeline）：

- **知识库 JSON**（`data/output/*/stage3_dryrun.json` / `stage3_kb.json`）：Alleys 聊天所用的剧情知识
- **全局角色 / 剧情弧**（`data/output/_global/`）：跨集角色画像和剧情线
- **声纹 / OCR / 场景数据**（`audio.json` / `visual.json` / `scenes.json`）

**不含**视频文件（`data/videos/`），需自行放入。

## License

MIT
