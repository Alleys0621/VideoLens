# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库工作时提供指引。

**重要约定：与用户的所有交流一律使用中文回复，不要切换到英文。**

## 项目概述

AlleysVid 是一个 AI 陪看智能体。用户选一集视频，Alleys（AI 搭子）陪你一起看，边看边聊剧情。支持流式语音对话（ASR 说话 → AI 回答 → TTS 播报）、用户系统、播放进度记忆、分层用户画像、中文会话标题。

技术栈：LangGraph + Qwen + Next.js + DashScope（ASR/TTS）+ Postgres。

## 架构与核心设计

### 用户理解层

- `src/agent/intent_router.py`
  - 强理解：`llm_route_intent()` 调用 **qwen-max**，输出 task / task_confidence / emotion / emotion_confidence / user_state。
  - 轻量路由：`route_intent()` 用 embedding 余弦匹配预定义意图目录，用于 deictic / meta / chitchat 快速分流，零额外 LLM 调用。
  - 保守回退：task_confidence < 0.6 时 `safe_task = chitchat`；emotion_confidence < 0.6 时 `safe_emotion = neutral`。

### 上下文白名单（Context Budget）

- `src/agent/context_builder.py`
  - 原则：**Context is a privilege, not a default.**
  - `CONTEXT_BUDGET` 按 task 决定哪些上下文段能进入 prompt。
  - `derive_watching_context()` 从当前播放时间派生轻量画面上下文（焦点角色、当前事件、当前 keyframe）。
  - 闲聊（chitchat）不给剧情/检索/记忆；问剧情（knowledge）才给检索事件和记忆。

### 分层画像

- L1 用户画像：`src/agent/profile_store.py` + `profile_updater.py::maybe_update_user_profile()`
  - 表 `user_profiles`：interaction_style / spoiler_tolerance / humor_level / engagement_motivation / alleys_attitude / confidence。
  - 每累计 2 条对话触发一次 `qwen3.7-flash` 增量更新。
  - `render_profile_overlay()` 无条件注入（V1.3.1 去掉 confidence 门控）。
- L2 作品画像：`profile_updater.py::maybe_update_show_profile()`
  - 表 `show_profiles`：favorite_characters / attention_characters / character_opinions / theme_preferences / disliked_elements / confidence。
  - 每轮全量注入 user prompt（V1.4 去意图路由）。

### 会话标题

- `src/agent/thread_title.py`
  - 首轮用户消息后异步生成 4–12 字中文标题，写入 `threads.custom_title`。
  - 使用 `qwen-turbo`，失败 fallback 截断首句。

### Checkpointer

- `src/server/checkpointer.py`：LangGraph 使用 `AsyncPostgresSaver`，对话 state 持久化到 Postgres。
- 首次连接时自动建 checkpoints / checkpoint_writes / checkpoint_blobs 表。

### 前端代理

- LangGraph SDK 请求走 `frontend/src/app/api/[..._path]/route.ts` 转发到后端 `:2024`。
- 自定义 thread 元数据端点放在 `frontend/src/app/api/chat-threads/*`，避免与 SDK 代理路由冲突。

## 启动

```bash
# 一键启动所有服务（Windows）
start.bat
```

服务端口：
- Frontend: http://localhost:3000
- Backend (LangGraph): http://localhost:2024
- ASR server (WebSocket): ws://localhost:9800/stream
- TTS server (WebSocket): ws://localhost:9801/
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
- **改后端 prompt / 人设后必须验证一条真实对话**：确认没有机器人味/幻觉
- **JSON 输出**：统一用 `src/core/helpers/json_utils.py::save_json`
- **Windows 中文路径**：`cv2.imwrite` 对中文路径失败，用 `_imwrite_unicode`
- **DASHSCOPE_API_KEY**：系统环境变量，不写入 `.env`
- **讯飞凭证**（`XFYUN_*`）：放在 `.env`
- **声纹组**：按 `data/videos/` 下作品名匹配 `pipeline.yaml` 的 `voiceprint_groups`
- **Context Budget**：新增 context 段必须先问“哪个 task 需要它”，不要默认注入
- **画像更新**：L1/L2 画像不每轮写，用 `PROFILE_UPDATE_THRESHOLD` 控制频率，避免 LLM 调用爆炸
- **Thread 标题**：异步生成，禁止阻塞流式回复
- **人设边界**：Alleys 不能假装看到画面；没给当前画面 context 时不描述表情/动作/眼神

### 代码规范（强制）

- **配置集中**：所有 env var / yaml 配置必经 `AppConfig`（`src/core/config.py::load_config`），业务代码禁止 `os.getenv` 直读
- **LLM 客户端**：统一用 `BaseLLMClient` / `QwenTextClient` / `QwenVLClient`；禁止业务代码直接 `import dashscope`（embedding 例外，但必须封装在 `retriever.py::_embed_texts` 内）
- **日志**：用 `from src.core.logging import get_logger; logger = get_logger(__name__)`；禁止 `print`（一次性调试可用，但不得 commit）
- **JSON 持久化**：文件 IO 统一 `save_json` / `load_json`（`src/core/helpers/json_utils.py`）；WebSocket/HTTP 响应消息可继续用 `json.dumps`
- **异常**：禁止裸 `except:` / `except Exception:`；必须 `except X as e:` + `logger.warning`/`.exception`
- **类型注解**：新式 `X | None`，禁止 `Optional[X]`
- **调试打点**：临时计时不得进生产 payload（如 `reasoning` 返回字段），必须用 env 开关（如 `VIDEOLENS_PERF`）门控
- **JSON 提取**：从 LLM 输出提取 JSON 用 `extract_json_obj`（`src/core/helpers/text_utils.py`），禁止手写 `re.search(r"\{...\}")`
- **注释精简（强制）**：禁止冗长注释。注释只写"为什么"（非常规理由），不写"是什么"（代码本身说明一切）。docstring 单行；禁止整段背景叙述、重复代码逻辑的解释、堆叠无信息量的分隔注释。长说明放 yaml 或设计文档，不放代码里。
- **代码生成（强制）**：生成/修改代码一律走工具规范化流程（先 Read 原文件 → Edit/Write 精确 diff），禁止凭空整段输出、禁止复制粘贴式乱改；每处改动需可被 git diff 校验。
- **改签名/返回类型（强制）**：改了函数签名或返回类型，必须 grep 全部调用方确认适配，禁止只改函数本身。典型坑：函数返回 bool，调用方还在 `n = f(); if n >= 阈值`（bool>=2 永远 False，副作用永远不发生）。
- **冒烟测到调用链末端（强制）**：冒烟不能只验证中间函数返回值对，必须验证**最终副作用真的发生**（mock 末端函数被调 / DB 真的写入 / 端到端跑通）。中间函数返回对了不代表调用方用对了——漏掉末端验证 = 偷懒 = 放 bug 进生产。

## 数据布局（已 gitignore）

```
data/
├── videos/{作品名}/{剧集}         # 输入视频
├── output/{video_dir}/            # 流水线产物 (audio.json / visual.json / stage3_dryrun.json ...)
└── output/_global/                # 跨集全局产物 (characters / global_arcs / character_profiles)
```

Postgres 业务表（由 `db/init.sql` + migrations 管理）：
- `users`：用户账号
- `user_profiles`：L1 跨会话画像（聊法 / 剧透 / 接梗）
- `show_profiles`：L2 作品级画像（喜欢角色 / 角色评价）
- `playback_progress`：用户 × 视频播放进度
- `threads`：会话元数据（含 `custom_title` 中文标题）
- `checkpoints` / `checkpoint_writes` / `checkpoint_blobs`：LangGraph state，由 `AsyncPostgresSaver.setup()` 自动创建

## 已搁置的实验路线

- Stage 1.5 LLM 说话人修正（纯文本方法不可行，信息论天花板）
- Stage 1 配置块（死配置，业务代码走 cli 参数）
