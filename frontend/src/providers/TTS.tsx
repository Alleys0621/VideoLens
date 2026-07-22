"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useStreamingTTS } from "@/hooks/useStreamingTTS";

/**
 * TTSProvider — 把 useStreamingTTS hook 单例化.
 *
 * 所有组件 (AssistantMessage 的喇叭按钮 + Thread 的自动朗读) 共享同一个
 * tts 实例, 互相打断 (点喇叭会 stop 自动朗读, 反之亦然).
 *
 * 后端走 src/agent/tts_server.py (常驻 ws :8001), 协议:
 *   start (预热 task) → text (LLM 增量) → finish (LLM 结束)
 *
 * 前端用 MediaSource + SourceBuffer 流式播放 mp3 chunks.
 */

type TTSInstance = ReturnType<typeof useStreamingTTS>;

const TTSContext = createContext<TTSInstance | undefined>(undefined);

export function TTSProvider({ children }: { children: ReactNode }) {
  const tts = useStreamingTTS();
  return (
    <TTSContext.Provider value={tts}>{children}</TTSContext.Provider>
  );
}

export function useTTSContext(): TTSInstance {
  const ctx = useContext(TTSContext);
  if (!ctx) {
    throw new Error("useTTSContext 必须在 TTSProvider 内部使用");
  }
  return ctx;
}
