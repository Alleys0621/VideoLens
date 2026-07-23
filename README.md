# AlleysVid — AI 陪看智能体

> **V1.0** — 系统成型的第一版代码

边看剧边聊天的 AI 陪看搭子。选一集视频，Alleys 陪你一起看，随时聊剧情、问角色、吐槽笑点。

## 功能

- **AI 陪看聊天**：选集后直接对话，Alleys 基于剧情知识库回答你的问题
- **流式语音对话**：说话给 Alleys 听（流式 ASR 实时转文字），Alleys 回答自动语音播报（流式 TTS）
- **用户系统**：手机号注册登录，每人独立的对话历史和记忆
- **播放进度记忆**：切走再回来，从上次的位置继续看
- **多设备访问**：本地或公网都能用（cloudflared 隧道）

## 快速开始

### 1. 环境要求

- **Python 3.12.x**（必须 < 3.13）
- **Node.js 18+** + **pnpm**（`npm install -g pnpm`）
- **ffmpeg**（系统 PATH 中可访问）

### 2. 克隆仓库

```bash
git clone https://github.com/Alleys0621/VideoLens.git
cd VideoLens
```

### 3. 安装 Python 依赖

```bash
uv venv .venv
.venv\Scripts\activate          # Windows
uv pip install -r requirements.txt
```

### 4. 安装前端依赖

```bash
cd frontend
pnpm install
cd ..
```

### 5. 配置环境变量

**系统环境变量**（必须）：
```
DASHSCOPE_API_KEY=你的阿里云百炼API Key
```

**项目根目录 `.env`**（讯飞声纹，Pipeline 用，不做 Pipeline 可跳过）：
```
XFYUN_APP_ID=你的讯飞AppID
XFYUN_API_KEY=你的讯飞APIKey
XFYUN_API_SECRET=你的讯飞APISecret
```

**前端 `frontend/.env.local`**（从示例复制）：
```bash
cp frontend/.env.local.example frontend/.env.local
```
默认值本地开发够用，按需修改。

### 6. 放入视频

视频文件放到 `data/videos/{作品名}/{剧集名}.mp4`，例如：
```
data/videos/家有儿女/第一季/第01集.mkv
data/videos/家有儿女/第二季/第001集.mkv
```

仓库自带知识库 JSON（角色、剧情、对白），**不需要重新跑 Pipeline**。但视频文件本身需自行提供。

### 7. 下载 cloudflared（可选，公网访问用）

从 [Cloudflare releases](https://github.com/cloudflare/cloudflared/releases) 下载 `cloudflared.exe`，放到 `.tools/` 目录。不需要公网访问可跳过。

### 8. 启动

```bash
# Windows 一键启动
start.bat
```

启动后会开 5 个窗口：
- Backend（LangGraph :2024）
- Frontend（Next.js :3000）
- ASR server（流式语音识别 :9800）
- TTS server（流式语音合成 :9801）
- cloudflared 隧道（公网访问，每次随机域名，需步骤 7）

浏览器打开 `http://localhost:3000`，注册登录后选一集开始。

## 技术栈

- **AI 对话**：LangGraph + Qwen（通义千问）
- **语音识别**：DashScope paraformer-realtime-v2（流式 WebSocket）
- **语音合成**：DashScope qwen-audio-3.0-tts-flash + longanhuan_v3.6（流式 WebSocket）
- **前端**：Next.js + TailwindCSS + Artplayer
- **用户存储**：LangGraph Store（无独立数据库）
- **公网隧道**：cloudflared quick tunnel

## 包含的数据

仓库自带以下数据，clone 后开箱即用（无需重新跑 Pipeline）：

- **用户数据库**（`.langgraph_api/store.pckl`）：已注册的用户账号
- **知识库 JSON**（`data/output/*/stage3_dryrun.json` / `stage3_kb.json`）：Alleys 聊天所用的剧情知识
- **全局角色 / 剧情弧**（`data/output/_global/`）：跨集角色画像和剧情线
- **声纹 / OCR / 场景数据**（`audio.json` / `visual.json` / `scenes.json`）

**不含**视频文件（`data/videos/`），需自行放入。

## License

MIT
