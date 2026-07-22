# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库工作时提供指引。

**重要约定：与用户的所有交流一律使用中文回复，不要切换到英文。**

## 项目概述

AlleysVid 是一个 AI 陪看智能体。用户选一集视频，Alleys（AI 搭子）陪你一起看，边看边聊剧情。支持流式语音对话（ASR 说话 → AI 回答 → TTS 播报）、用户系统、播放进度记忆。

技术栈：LangGraph + Qwen + Next.js + DashScope（ASR/TTS）。

## 启动

```bash
# 一键启动所有服务（Windows）
start.bat
```

服务端口：
- Frontend: http://localhost:3000
- Backend (LangGraph): http://localhost:2024
- ASR server (WebSocket): ws://localhost:8000/stream
- TTS server (WebSocket): ws://localhost:8001/
- cloudflared 隧道（公网，随机域名）

## 常用命令

```bash
# 运行 Pipeline（处理视频）
python -m src.app.main run "家有儿女/第一季/第01集"

# 跑单个阶段
python -m src.app.main run "家有儿女/第一季/第01集" --stage 2

# 前端开发
cd frontend && pnpm dev

# 前端类型检查
cd frontend && npx tsc --noEmit

# 认证 e2e 测试
cd frontend && node scripts/test-auth.mjs

# 多用户隔离测试
cd frontend && node scripts/test-thread-isolation.mjs
```

## 约定

- **包前缀**：`src.`（如 `from src.core.config import get_config`）
- **中文输出**：所有面向用户的字符串、日志、评估报告都使用中文
- **改前端代码后必须验证编译**：`npx tsc --noEmit`
- **JSON 输出**：统一用 `src/core/helpers/json_utils.py::save_json`
- **Windows 中文路径**：`cv2.imwrite` 对中文路径失败，用 `_imwrite_unicode`
- **DASHSCOPE_API_KEY**：系统环境变量，不写入 `.env`
- **讯飞凭证**（`XFYUN_*`）：放在 `.env`
- **声纹组**：按 `data/videos/` 下作品名匹配 `pipeline.yaml` 的 `voiceprint_groups`

## 数据布局（已 gitignore）

```
data/
├── videos/{作品名}/{剧集}         # 输入视频
├── output/{video_dir}/            # 流水线产物 (audio.json / visual.json / stage3_dryrun.json ...)
└── output/_global/                # 跨集全局产物 (characters / global_arcs / character_profiles)
```

## 已搁置的实验路线

- Stage 1.5 LLM 说话人修正（纯文本方法不可行，信息论天花板）
- Stage 1 配置块（死配置，业务代码走 cli 参数）
