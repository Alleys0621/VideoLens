"""AlleysVid — AI 陪看智能体 (边看剧边聊)

顶层模块:
  agent/    在线对话 (companion/画像/记忆/语音)
  pipeline/ 离线视频处理 (Stage1-3 → KB)
  server/   LangGraph graph + Postgres checkpointer
  app/      Pipeline CLI
  core/     基础设施 (config/llm/logging/helpers)
  eval/ voiceprint/
"""
