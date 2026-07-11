"use client";

import { useState, useRef, useCallback } from "react";

/**
 * TTS 播放 hook: 调用 /api/tts (CosyVoice 后端) 把文字转语音并播放.
 * speak(text) → POST 音频 → Audio 元素播放; stop() 停止.
 *
 * 用于 AI 回复的"朗读"按钮 (用户主动点, 不自动播避免扰民).
 */
export function useTTS() {
  const [speaking, setSpeaking] = useState(false);
  const [loading, setLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const speak = useCallback(async (text: string) => {
    if (!text.trim()) return;
    // 停止之前的播放
    audioRef.current?.pause();
    setLoading(true);
    setSpeaking(false);
    try {
      const resp = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await resp.json();
      if (data.audio) {
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`);
        audioRef.current = audio;
        audio.onplay = () => {
          setSpeaking(true);
          setLoading(false);
        };
        audio.onended = () => setSpeaking(false);
        audio.onerror = () => {
          setSpeaking(false);
          setLoading(false);
        };
        await audio.play();
      } else {
        setLoading(false);
      }
    } catch {
      setLoading(false);
      setSpeaking(false);
    }
  }, []);

  const stop = useCallback(() => {
    audioRef.current?.pause();
    setSpeaking(false);
  }, []);

  return { speak, stop, speaking, loading };
}
