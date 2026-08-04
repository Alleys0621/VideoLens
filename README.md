# AlleysVid — AI 陪看智能体

> **V1.4.2** — 观看状态上报机制（前端持续上报视频坐标，agent 检索拿到最新播放位置）
> 完整版本历史见 [VERSION.md](./VERSION.md)

边看剧边聊天的 AI 陪看搭子。选一集视频，Alleys 陪你一起看，随时聊剧情、问角色、吐槽笑点。

## 功能

- **AI 陪看聊天**：选集后直接对话，Alleys 基于剧情知识库回答
- **流式语音对话**：流式 ASR 实时转文字 + 流式 TTS 自动播报
- **用户系统**：手机号注册登录，每人独立的对话历史和记忆
- **播放进度记忆**：切走再回来从上次位置继续
- **观看状态上报**：前端定时上报视频坐标，agent 检索拿到最新播放位置（V1.4.2）
- **分层用户画像**：L1 跨会话聊法偏好 + L2 单部作品角色偏好，记忆跟着人走
- **中文会话标题**：首条消息后自动起 4–12 字中文标题
- **人设**：INFJ/ENFP 混合人格 + 反机器人味 + 反幻觉约束，更像真人搭子
- **多设备访问**：本地或公网（cloudflared 隧道）
- **数据持久化**：Postgres 存用户/画像/进度/thread 元数据/对话 state，重启不丢

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

或手动下载（建议手动下载哈，因为要挂梯子的，不挂梯子下载巨慢）：访问 [cloudflare/cloudflared releases](https://github.com/cloudflare/cloudflared/releases)，下载 `cloudflared-windows-amd64.exe`，重命名为 `cloudflared.exe` 放到 `.tools/` 目录下。

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

## 日常更新（每次 `git pull` 之后必做）

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
