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

### 环境要求

- Python 3.12.x
- Node.js 18+（pnpm）
- ffmpeg

### 安装

```bash
# Python 后端
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# 前端
cd frontend
pnpm install
```

### 配置

1. 设置系统环境变量 `DASHSCOPE_API_KEY`（阿里云百炼）
2. 讯飞声纹凭证写入 `.env`（`XFYUN_*` 字段）
3. 前端配置参考 `frontend/.env.local.example`

### 启动

```bash
# Windows 一键启动（双击 start.bat 或命令行运行）
start.bat
```

启动后会开 5 个窗口：
- Backend（LangGraph :2024）
- Frontend（Next.js :3000）
- ASR server（流式语音识别 :8000）
- TTS server（流式语音合成 :8001）
- cloudflared 隧道（公网访问，每次随机域名）

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

**不含**视频文件（`data/videos/`），需自行放入视频到 `data/videos/` 目录。

## License

MIT
