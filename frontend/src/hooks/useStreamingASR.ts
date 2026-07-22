"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { toast } from "sonner";

/**
 * 流式 ASR hook: 边录边识别, 边显示文本.
 *
 * 架构:
 *   AudioWorklet (16kHz mono Float32 PCM)
 *     ↓ port.postMessage (每 200ms 一个 chunk)
 *   主线程转 Int16 PCM bytes
 *     ↓ WebSocket binary frame
 *   Python asr_server (localhost:8000/stream, 常驻)
 *     ↓ DashScope paraformer-realtime-v2 streaming
 *   识别结果 partial/final 通过 WebSocket 回推
 *
 * 防音频回声 (浏览器 AEC 对媒体元素不可靠):
 *   录音开始时调 onPauseOthers (上层负责停 TTS + 暂停视频)
 *   录音结束时调 onResumeOthers (上层恢复视频)
 *
 * 用法:
 *   const recorder = useStreamingASR({
 *     onPauseOthers: () => { tts.stop(); videoControlRef.current?.pause(); },
 *     onResumeOthers: () => { videoControlRef.current?.resume(); },
 *   });
 *   await recorder.start();  // 边录边识别, 边显示
 *   const text = await recorder.stop();  // 拿最终文本
 */

type ASRStatus = "idle" | "starting" | "recording" | "stopping" | "error";

interface UseStreamingASROptions {
  /** 录音开始时调用 (停 TTS + 暂停视频, 防回声) */
  onPauseOthers?: () => void;
  /** 录音结束时调用 (恢复视频播放) */
  onResumeOthers?: () => void;
}

interface UseStreamingASRReturn {
  start: () => Promise<void>;
  stop: () => Promise<string>;
  isRecording: boolean;
  status: ASRStatus;
  partialText: string;
  finalText: string;
  error: string | null;
}

/**
 * ASR WebSocket 地址.
 *
 * 智能选择:
 *   - 显式 env NEXT_PUBLIC_ASR_WS_URL 优先 (生产 wss://agent.alleysvid.xyz/asr)
 *   - 页面 https:// → 必须用 wss:// (浏览器禁止 mixed content)
 *     默认 wss://当前host/asr (需要 nginx + frp 配 wss 反代到本地 :8000)
 *   - 页面 http:// → 用 ws://localhost:8000/stream (本地开发)
 */
function getAsrWsUrl(): string {
  if (process.env.NEXT_PUBLIC_ASR_WS_URL) {
    return process.env.NEXT_PUBLIC_ASR_WS_URL;
  }
  if (typeof window === "undefined") return "ws://localhost:8000/stream";
  const pageProto = window.location.protocol;
  const pageHost = window.location.hostname;
  if (pageProto === "https:") {
    // HTTPS 页面必须用 wss://, ws:// 会被 mixed content 阻止
    return `wss://${pageHost}/asr`;
  }
  // 本地开发: http://localhost:3000 → 直连本地 ASR server
  return "ws://127.0.0.1:9800/stream";
}

export function useStreamingASR(
  options: UseStreamingASROptions = {},
): UseStreamingASRReturn {
  const [status, setStatus] = useState<ASRStatus>("idle");
  const [partialText, setPartialText] = useState("");
  const [finalText, setFinalText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const partialRef = useRef("");
  const finalRef = useRef("");

  // 用 ref 持有最新的 callback, 避免 start/stop 重建 (依赖空数组)
  const onPauseOthersRef = useRef(options.onPauseOthers);
  const onResumeOthersRef = useRef(options.onResumeOthers);
  useEffect(() => {
    onPauseOthersRef.current = options.onPauseOthers;
    onResumeOthersRef.current = options.onResumeOthers;
  }, [options.onPauseOthers, options.onResumeOthers]);

  // 转 Float32 → Int16 (PCM s16le, DashScope 要求)
  const float32ToInt16 = useCallback((float32: Float32Array): ArrayBuffer => {
    const buffer = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  }, []);

  const cleanup = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.port.close();
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    if (status === "recording" || status === "starting") return;
    setStatus("starting");
    setError(null);
    partialRef.current = "";
    finalRef.current = "";
    setPartialText("");
    setFinalText("");

    try {
      // 关键: 录音前先停 TTS + 暂停视频, 防止回声 (浏览器 AEC 对媒体元素不可靠)
      onPauseOthersRef.current?.();

      // 1. 建立 WebSocket
      const wsUrl = getAsrWsUrl();
      console.log("[ASR] connecting to:", wsUrl);
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error("ASR 服务连接超时, 请确认 asr_server 已启动 (port 9800)")),
          5000,
        );
        ws.onopen = () => {
          clearTimeout(timer);
          console.log("[ASR] WebSocket connected");
          resolve();
        };
        ws.onerror = (ev) => {
          clearTimeout(timer);
          console.error("[ASR] WebSocket error:", ev);
          reject(
            new Error(
              `ASR 服务连接失败 (${wsUrl}). ${
                wsUrl.startsWith("wss://")
                  ? "HTTPS 部署需配 nginx/frp 反代 wss → 本地 :8000"
                  : "请确认 asr_server 已启动"
              }`,
            ),
          );
        };
      });

      let msgCount = 0;
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          msgCount++;
          if (msgCount <= 5 || msg.type === "final") {
            console.log(`[ASR] ws msg #${msgCount}:`, msg);
          }
          if (msg.type === "partial") {
            partialRef.current = msg.text;
            setPartialText(msg.text);
          } else if (msg.type === "final") {
            finalRef.current = (finalRef.current + " " + msg.text).trim();
            setFinalText(finalRef.current);
            partialRef.current = "";
            setPartialText("");
          } else if (msg.type === "error") {
            setError(msg.message || "ASR 错误");
            toast.error(`识别错误: ${msg.message || "未知"}`);
          } else if (msg.type === "ready") {
            console.log("[ASR] DashScope session ready");
          }
        } catch (err) {
          console.warn("[ASR] parse ws message failed", err);
        }
      };

      // 2. AudioContext + AudioWorklet
      const AC = window.AudioContext || (window as any).webkitAudioContext;
      const audioContext = new AC({ sampleRate: 48000 });
      audioContextRef.current = audioContext;

      await audioContext.audioWorklet.addModule("/audio-processor.js");

      // 3. 麦克风 (echoCancellation 提升识别率, 但不依赖它消除媒体回声)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      const source = audioContext.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(audioContext, "audio-processor");
      workletNodeRef.current = worklet;

      // 4. AudioWorklet 输出 → 主线程转 Int16 → WebSocket 发送
      let pcmChunkCount = 0;
      worklet.port.onmessage = (e) => {
        const float32 = e.data as Float32Array;
        const pcm16 = float32ToInt16(float32);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(pcm16);
          pcmChunkCount++;
          if (pcmChunkCount === 1) {
            console.log(
              `[ASR] first PCM chunk sent: ${pcm16.byteLength} bytes, samples=${float32.length}`,
            );
          } else if (pcmChunkCount === 10 || pcmChunkCount === 50) {
            console.log(`[ASR] sent ${pcmChunkCount} PCM chunks so far`);
          }
        }
      };

      source.connect(worklet);
      // 注意: worklet 不连 destination, 否则会有反馈啸叫

      console.log("[ASR] recording started, sampleRate=48000");
      setStatus("recording");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("error");
      toast.error(`录音启动失败: ${msg}`);
      cleanup();
      // 失败时恢复视频 (没真正开始录)
      onResumeOthersRef.current?.();
    }
  }, [status, cleanup, float32ToInt16]);

  const stop = useCallback(async (): Promise<string> => {
    if (status !== "recording") return finalRef.current;
    setStatus("stopping");

    // 通知 ASR server 停止 (让它把最后的 partial 转成 final)
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "stop" }));
    }

    // 等 0.5s 让最后的 final 回来
    await new Promise((r) => setTimeout(r, 500));

    cleanup();
    setStatus("idle");

    // 合并 final + 最后的 partial (如果还有)
    const result = (finalRef.current + " " + partialRef.current).trim();

    // 录音结束 → 恢复视频
    onResumeOthersRef.current?.();

    return result;
  }, [status, cleanup]);

  // 卸载时清理
  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  return {
    start,
    stop,
    isRecording: status === "recording",
    status,
    partialText,
    finalText,
    error,
  };
}
