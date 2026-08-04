# Frontend Src

Next.js App Router (agent-chat-ui 改造)

## app/ — 路由
- `(auth)/` 登录 / 注册 / 找回密码 (路由组, 不进 URL)
- `api/`
  - `[..._path]/` LangGraph SDK 代理 → :2024
  - `auth/` 认证端点 (账号 / 短信 / 微信 / 改密)
  - `chat-threads/` 会话元数据 (避开 SDK 代理路由)
  - `playback/` `watching/` 播放进度 / 观看状态
  - `video/` `videos/` 视频资源代理
- `layout.tsx` / `page.tsx` 根布局 / 首页

## components/
- `video-player/` 播放器 + 关键帧跳转 (useKeyframeSeek)
- `thread/` 对话 + 消息渲染 (ai / human / reasoning)
- `icons/` `ui/` `layout/`

## hooks/
- `useStreamingASR` / `useStreamingTTS` 流式语音
- `useKeyframeSeek` 关键帧跳转 (解析 reasoning.keyframes)
- `useAbsoluteApiUrl` API URL (公网 / 本地切换)

## lib/
- `auth.ts` / `jwt.ts` 认证 (NextAuth + JWT)
- `db.ts` Postgres DAO (pg)
- `langgraph-client.ts` LangGraph SDK 客户端

## providers/ — React Context
- `Auth` / `Stream` / `TTS` / `Thread` / `client`

## middleware.ts
路由守卫 (登录态)
