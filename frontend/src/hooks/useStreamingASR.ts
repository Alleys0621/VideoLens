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
 *   Python asr_server (0.0.0.0:9800/stream, 常驻)
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
 *   - 显式 env NEXT_PUBLIC_ASR_WS_URL 优先
 *   - https 页面 → wss (mixed content 限制); 用当前 hostname + :9800 直连 ASR server (已启用 wss)
 *   - http 页面 → ws (本机/局域网); 同样用 hostname + :9800
 */
function getAsrWsUrl(): string {
  if (process.env.NEXT_PUBLIC_ASR_WS_URL) {
    return process.env.NEXT_PUBLIC_ASR_WS_URL;
  }
  if (typeof window === "undefined") return "ws://localhost:9800/stream";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const hostname = window.location.hostname;
  return `${proto}://${hostname}:9800/stream`;
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
  // worklet (SecureContext) 或 script processor (HTTP+IP fallback), 统一用 AudioNode 持有
  const audioNodeRef = useRef<AudioNode | null>(null);
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
    if (audioNodeRef.current) {
      // worklet 才有 port, script processor 没有
      const node = audioNodeRef.current as AudioNode & { port?: { close?: () => void } };
      try { node.port?.close?.(); } catch { /* noop */ }
      try { node.disconnect(); } catch { /* noop */ }
      audioNodeRef.current = null;
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
                  ? "公网部署需另配 wss 反代 (cloudflared named tunnel / nginx) 转发到本地 :9800"
                  : "请确认 asr_server 已启动且监听 0.0.0.0"
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

      // 2. AudioContext (audioWorklet 在 SecureContext 下才有, HTTP+IP 时为 undefined)
      const AC = window.AudioContext || (window as any).webkitAudioContext;
      const audioContext = new AC({ sampleRate: 48000 });
      audioContextRef.current = audioContext;

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

      // PCM 发送 (worklet / script 共用)
      let pcmChunkCount = 0;
      const sendPcm = (float32: Float32Array) => {
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

      if (audioContext.audioWorklet) {
        // SecureContext: AudioWorklet (独立线程, 不阻塞主线程)
        await audioContext.audioWorklet.addModule("/audio-processor.js");
        const worklet = new AudioWorkletNode(audioContext, "audio-processor");
        worklet.port.onmessage = (e) => sendPcm(e.data as Float32Array);
        source.connect(worklet);
        // 不连 destination, 否则反馈啸叫
        audioNodeRef.current = worklet;
        console.log("[ASR] using AudioWorklet (SecureContext)");
      } else {
        // HTTP+IP fallback: ScriptProcessorNode (主线程, deprecated 但全浏览器支持)
        // 性能足够: 200ms 一次 ~3200 samples, 主线程开销 < 1ms
        const node = audioContext.createScriptProcessor(2048, 1, 1);
        let buf = new Float32Array(0);
        const ratio = 48000 / 16000;
        const flushThreshold = 3200; // 200ms @ 16kHz
        node.onaudioprocess = (e: AudioProcessingEvent) => {
          const input = e.inputBuffer.getChannelData(0);
          // 与 audio-processor.js 完全一致的 downsample 逻辑 (nearest 采样)
          const outLen = Math.floor(input.length / ratio);
          const down = new Float32Array(outLen);
          for (let i = 0; i < outLen; i++) {
            down[i] = input[Math.floor(i * ratio)];
          }
          const merged = new Float32Array(buf.length + down.length);
          merged.set(buf, 0);
          merged.set(down, buf.length);
          while (merged.length >= flushThreshold) {
            sendPcm(merged.slice(0, flushThreshold));
            buf = merged.slice(flushThreshold);
            return; // 等下一个 process 事件
          }
          buf = merged;
        };
        source.connect(node);
        // ScriptProcessor 必须连 destination 才触发 onaudioprocess;
        // 我们没写 outputBuffer, 实际无声
        node.connect(audioContext.destination);
        audioNodeRef.current = node;
        console.warn(
          "[ASR] using deprecated ScriptProcessorNode (non-SecureContext, http+IP). " +
          "For better perf, use chrome://flags #unsafely-treat-insecure-origin-as-secure " +
          "or access via https.",
        );
      }

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
