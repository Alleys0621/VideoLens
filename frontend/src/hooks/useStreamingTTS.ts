"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { toast } from "sonner";

/**
 * useStreamingTTS — qwen-audio-3.0-tts-flash (longanhuan_v3.6) 真正流式朗读.
 *
 * ============================================================================
 * 完整调用链 (浏览器 → 本地 tts_server → DashScope ws)
 * ============================================================================
 *
 *   [浏览器] useStreamingTTS
 *      │  ws.send {type: "start"}              (建立 DashScope task)
 *      │  ws.send {type: "text", text}          (LLM 增量, 多次)
 *      │  ws.send {type: "finish"}              (LLM 结束)
 *      ▼
 *   [本地 tts_server] ws://localhost:8001/      (src/agent/tts_server.py)
 *      │  双向桥接前端 ws ↔ DashScope ws
 *      ▼
 *   [DashScope] qwen-audio-3.0-tts-flash
 *      │  边收 continue-task 边合成
 *      │  推 result-generated (二进制 mp3 chunk)
 *      ▼
 *   [本地 tts_server] 原样转发二进制 → 前端 ws
 *      ▼
 *   [浏览器] MediaSource + SourceBuffer.appendBuffer
 *      │  边收 mp3 chunk 边追加到 audio 流
 *      │  <audio> 元素实时播放 (首字延迟 ~400ms after 发 text)
 *      ▼
 *   [浏览器] 同时累积 chunks → 完整 Blob → 缓存 (再次点击秒播)
 *
 * ============================================================================
 * 性能 (实测, 详见 tts_server 测试日志)
 * ============================================================================
 *
 *   固定开销 (run-task → task-started): ~650-745ms
 *   发 text 后首字 mp3: ~332-527ms
 *   整体端到端首字: 1060-1272ms
 *
 * 优化: 前端在用户点发送瞬间调 start() (预热 task), LLM 出 token 时
 *      ready 已回来, 直接 feedText → 首字 ~400ms 到达.
 *
 * ============================================================================
 * 用法
 * ============================================================================
 *
 * 1. 自动朗读 (LLM 边推边 TTS):
 *    const tts = useStreamingTTS();
 *    await tts.start();                  // 用户点发送时调, 预热
 *    tts.feedText("你好");                // LLM 增量
 *    tts.feedText("呀,我是");             // LLM 继续推
 *    tts.feedText("Alleys。");
 *    await tts.finish();                 // LLM 结束, 等合成完成
 *
 * 2. 手动朗读 (整段):
 *    await tts.speak("完整一段文字");      // 自动 start/feedText/finish
 *
 * 3. 缓存: speak/feedText 完成后, 所有 mp3 chunks 累积成 Blob URL 存到
 *    messageCache (key=完整文本). 再次点击同一条消息的喇叭, 直接从缓存
 *    new Audio(blobUrl).play() 秒播, 不再合成.
 */

/** 获取 tts_server ws URL (开发直连, 生产 wss 反代) */
function getTtsWsUrl(): string {
  if (process.env.NEXT_PUBLIC_TTS_WS_URL) {
    return process.env.NEXT_PUBLIC_TTS_WS_URL;
  }
  if (typeof window === "undefined") return "ws://127.0.0.1:9801/";
  const pageProto = window.location.protocol;
  const pageHost = window.location.hostname;
  if (pageProto === "https:") {
    // 公网 (cloudflared): TTS ws 不走 cloudflared (只代理 :3000 HTTP).
    // 尝试用当前 host 的同端口 ws (如果后端配了 wss 反代则能用, 否则本地才有 ws).
    return `wss://${pageHost}/tts-ws`;
  }
  return "ws://127.0.0.1:9801/";
}

/** message-level 缓存: 文本 → 已合成好的 Blob URL. 跨 hook 实例共享. */
const messageCache = new Map<string, string>();

/** MSE SourceBuffer 一次最多缓冲多少字节后开始主动清理 (避免 QuotaExceeded) */
const MSE_BUFFER_LIMIT_BYTES = 5 * 1024 * 1024; // 5MB

interface UseStreamingTTSReturn {
  /** 建立 ws + 发 start, 等 ready. messageId 绑定到具体消息 (喇叭动画). */
  start: (messageId?: string) => Promise<void>;
  /** LLM 增量喂文本. 可多次调, 内部累积发给后端. */
  feedText: (delta: string) => void;
  /** 发 finish, 等 complete. LLM stream 结束时调. */
  finish: () => Promise<void>;
  /** 整段朗读 (手动点喇叭): start + feedText(整段) + finish. 自动缓存. */
  speak: (text: string, messageId?: string) => Promise<void>;
  /** 立即停止 + 清理 (用户中断 / 切对话). */
  stop: () => void;
  /** 是否在播 */
  speaking: boolean;
  /** 当前正在播放的消息 id (null = 未绑定/未播放) */
  speakingMessageId: string | null;
  /** 是否在等 ready / 合成中 */
  loading: boolean;
}

export function useStreamingTTS(): UseStreamingTTSReturn {
  const [speaking, setSpeaking] = useState(false);
  const [loading, setLoading] = useState(false);
  // 当前正在播放的消息 id, 用于让具体哪条消息的喇叭图标动起来
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaSourceRef = useRef<MediaSource | null>(null);
  const sourceBufferRef = useRef<SourceBuffer | null>(null);

  // MSE appendBuffer 队列 (上一次 appendBuffer 必须等 updateend)
  const mseQueueRef = useRef<Uint8Array[]>([]);
  const msePendingRef = useRef(false);
  const eosFlagRef = useRef(false);

  // 缓存累积: 一段会话的所有 mp3 chunk, 完成时拼成 Blob
  const collectedChunksRef = useRef<Uint8Array[]>([]);
  const fullTextRef = useRef<string>(""); // 当前会话的完整文本 (用于 cache key)

  // 状态标记
  const startedRef = useRef(false);
  const readyRef = useRef(false);
  const finishedRef = useRef(false);
  const finishWaiterRef = useRef<(() => void) | null>(null);

  // 关键: LLM 增量可能比 ready 早到 (handleSubmit 预热后 ~500ms LLM 出 token,
  // 但 ready 要 ~700ms). 此时 feedText 不能丢弃 delta, 必须缓存等 ready 后 flush.
  const pendingTextRef = useRef("");
  // finish 也可能在 ready 之前调 (LLM 输出很快), 同样缓存
  const finishPendingRef = useRef(false);

  /* ------------------------------------------------------------------ */
  /* MSE: 把一个 mp3 chunk 追加到 SourceBuffer, 维护 appendBuffer 队列    */
  /* ------------------------------------------------------------------ */
  const enqueueMseChunk = useCallback((data: Uint8Array) => {
    // 累积给缓存 (无论 MSE 是否就绪, 都要存)
    collectedChunksRef.current.push(data);

    const sb = sourceBufferRef.current;
    // MSE 还没就绪 (sourceopen 没触发), 缓存到队列等 sourceopen 后 flush
    // 否则之前直接 return 会丢弃 chunk → MediaSource 无数据 → audio error
    if (!sb || sb.updating || msePendingRef.current) {
      mseQueueRef.current.push(data);
      return;
    }
    msePendingRef.current = true;
    try {
      sb.appendBuffer(data);
    } catch (e) {
      console.error("[useStreamingTTS] appendBuffer failed", e);
      msePendingRef.current = false;
    }
  }, []);

  const pumpMseQueue = useCallback(() => {
    const sb = sourceBufferRef.current;
    if (!sb || sb.updating) return;
    if (mseQueueRef.current.length === 0) {
      // 队列空, 如果已标记 eos 就 endOfStream
      if (eosFlagRef.current) {
        try {
          mediaSourceRef.current?.endOfStream();
        } catch {
          /* noop */
        }
      }
      return;
    }
    const next = mseQueueRef.current.shift()!;
    msePendingRef.current = true;
    try {
      sb.appendBuffer(next);
    } catch (e) {
      console.error("[useStreamingTTS] appendBuffer (pump) failed", e);
      msePendingRef.current = false;
      pumpMseQueue(); // 递归处理下一个
    }
  }, []);

  /* ------------------------------------------------------------------ */
  /* 创建 MSE audio 元素 + 挂 SourceBuffer                                */
  /* ------------------------------------------------------------------ */
  const setupAudio = useCallback(() => {
    // 清理旧的 audio (注意: 设 src="" 会触发 onerror 误报, 用 flag 屏蔽)
    if (audioRef.current) {
      const old = audioRef.current;
      old.onended = null;
      old.onerror = null;
      old.onplay = null;
      old.onpause = null;
      old.pause();
      try {
        old.src = "";
      } catch {
        /* noop */
      }
    }
    // 旧的 MediaSource 直接释放引用, 不主动 endOfStream (会触发各种 race)
    // 让 GC 自然回收
    if (mediaSourceRef.current) {
      try {
        if (mediaSourceRef.current.readyState === "open") {
          mediaSourceRef.current.endOfStream();
        }
      } catch {
        /* noop */
      }
    }
    sourceBufferRef.current = null; // 关键: 清掉旧引用, 否则 enqueueMseChunk 用错 sb

    // 重置 MSE 队列状态
    mseQueueRef.current = [];
    msePendingRef.current = false;
    eosFlagRef.current = false;

    const audio = new Audio();
    const ms = new MediaSource();
    const objectUrl = URL.createObjectURL(ms);
    audio.src = objectUrl;
    audioRef.current = audio;
    mediaSourceRef.current = ms;

    ms.addEventListener("sourceopen", () => {
      try {
        const sb = ms.addSourceBuffer("audio/mpeg");
        sb.mode = "sequence";
        sourceBufferRef.current = sb;
        let firstChunkAppended = false;
        sb.addEventListener("updateend", () => {
          msePendingRef.current = false;
          // 第一个 chunk 处理完才调 play (避免空数据触发 audio error)
          if (!firstChunkAppended) {
            firstChunkAppended = true;
            audio.play().catch((e) => {
              console.warn("[useStreamingTTS] play blocked:", e);
            });
          }
          // 主动清理超限 buffer (避免内存爆炸)
          try {
            if (sb.buffered.length > 0 && sb.buffered.end(0) > 60) {
              sb.remove(0, sb.buffered.end(0) - 30);
              return; // remove 也会触发 updateend, 不立即 pump
            }
          } catch {
            /* noop */
          }
          pumpMseQueue();
        });
        // sourceopen 后立即 flush 预先到达的 chunks (源 buffer 就绪前累积的)
        pumpMseQueue();
      } catch (e) {
        console.error("[useStreamingTTS] addSourceBuffer failed", e);
        toast.error("音频初始化失败");
      }
    });

    audio.onplay = () => setSpeaking(true);
    audio.onpause = () => {
      if (audio.ended) setSpeaking(false);
    };
    audio.onended = () => { setSpeaking(false); setSpeakingMessageId(null); };
    audio.onerror = () => {
      // 清理时 src="" 会触发 onerror, 此时 src 为空直接忽略
      if (!audio.src) return;
      const err = audio.error;
      console.error(
        "[useStreamingTTS] audio error",
        err
          ? {
              code: err.code,
              message:
                err.code === 1
                  ? "ABORTED"
                  : err.code === 2
                    ? "NETWORK"
                    : err.code === 3
                      ? "DECODE"
                      : err.code === 4
                        ? "SRC_NOT_SUPPORTED"
                        : "unknown",
              ms_state: mediaSourceRef.current?.readyState,
              sb_exists: !!sourceBufferRef.current,
            }
          : "(null)",
      );
      setSpeaking(false);
    };
  }, [pumpMseQueue]);

  /* ------------------------------------------------------------------ */
  /* 清理当前会话的所有资源                                               */
  /* ------------------------------------------------------------------ */
  const cleanup = useCallback((opts?: { keepAudioPlaying?: boolean }) => {
    // ws
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        /* noop */
      }
      wsRef.current = null;
    }
    // MSE 状态 — keepAudioPlaying 时不清 eosFlag (让 pumpMseQueue 能正常 endOfStream)
    mseQueueRef.current = [];
    msePendingRef.current = false;
    if (!opts?.keepAudioPlaying) {
      eosFlagRef.current = false;
    }
    // 会话状态
    startedRef.current = false;
    readyRef.current = false;
    finishedRef.current = false;
    finishWaiterRef.current = null;
    // keepAudioPlaying: 合成完成但音频还在 MSE buffer 里播, 不停 speaking
    // 让 audio.onended 控制 speaking=false
    if (!opts?.keepAudioPlaying) {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      setSpeaking(false);
      setSpeakingMessageId(null);
    }
    setLoading(false);
  }, []);

  /* ------------------------------------------------------------------ */
  /* 处理 ws 收到的消息 (二进制 mp3 chunk / JSON 事件)                   */
  /* ------------------------------------------------------------------ */
  const handleWsMessage = useCallback(
    (event: MessageEvent) => {
      if (event.data instanceof ArrayBuffer) {
        // mp3 二进制 chunk, 追加到 MSE
        enqueueMseChunk(new Uint8Array(event.data));
        return;
      }
      if (event.data instanceof Blob) {
        // 某些浏览器把二进制包成 Blob
        event.data.arrayBuffer().then((buf) => {
          enqueueMseChunk(new Uint8Array(buf));
        });
        return;
      }
      // JSON 消息
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "ready") {
          readyRef.current = true;
          setLoading(false);
          // 关键: flush 预热期间累积的 LLM 增量 (避免 delta 在 ready 之前到达被丢弃)
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            if (pendingTextRef.current) {
              console.log(
                `[useStreamingTTS] flushing pending text on ready: ${pendingTextRef.current.length} chars`,
              );
              ws.send(
                JSON.stringify({ type: "text", text: pendingTextRef.current }),
              );
              pendingTextRef.current = "";
            }
            // 如果 LLM 已结束 (finish 在 ready 之前调), 现在发 finish
            if (finishPendingRef.current) {
              finishPendingRef.current = false;
              console.log("[useStreamingTTS] flushing pending finish on ready");
              ws.send(JSON.stringify({ type: "finish" }));
            }
          }
        } else if (msg.type === "complete") {
          // 合成完成
          eosFlagRef.current = true;
          // 如果 MSE 队列空了, 立即 endOfStream
          if (!msePendingRef.current && mseQueueRef.current.length === 0) {
            try {
              mediaSourceRef.current?.endOfStream();
            } catch {
              /* noop */
            }
          } else {
            pumpMseQueue();
          }
          // 保存到 message-level 缓存
          if (
            collectedChunksRef.current.length > 0 &&
            fullTextRef.current
          ) {
            try {
              const blob = new Blob(collectedChunksRef.current, {
                type: "audio/mpeg",
              });
              const url = URL.createObjectURL(blob);
              // 旧缓存先释放
              const old = messageCache.get(fullTextRef.current);
              if (old) URL.revokeObjectURL(old);
              messageCache.set(fullTextRef.current, url);
            } catch (e) {
              console.error("[useStreamingTTS] cache save failed", e);
            }
          }
          finishedRef.current = true;
          finishWaiterRef.current?.();
          // 合成完成, 但音频还在 MSE buffer 里播放, 不停 speaking
          cleanup({ keepAudioPlaying: true });
        } else if (msg.type === "error") {
          console.error("[useStreamingTTS] server error", msg.message);
          toast.error(`TTS 错误: ${msg.message || "未知"}`);
          cleanup();
        }
      } catch (err) {
        console.warn("[useStreamingTTS] parse ws message failed", err);
      }
    },
    [enqueueMseChunk, pumpMseQueue, cleanup],
  );

  /* ------------------------------------------------------------------ */
  /* 公开 API                                                            */
  /* ------------------------------------------------------------------ */

  /** 建立 ws + 发 start, 等 ready. messageId 用于绑定到具体消息. */
  const start = useCallback(async (messageId?: string) => {
    if (startedRef.current) return;
    startedRef.current = true;
    setLoading(true);
    if (messageId) setSpeakingMessageId(messageId);

    // 重置会话级状态
    readyRef.current = false;
    finishedRef.current = false;
    eosFlagRef.current = false;
    collectedChunksRef.current = [];
    fullTextRef.current = "";
    pendingTextRef.current = "";
    finishPendingRef.current = false;

    // 建 MSE audio 元素
    setupAudio();

    // 建 ws
    const url = getTtsWsUrl();
    console.log("[useStreamingTTS] connecting:", url);
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onmessage = handleWsMessage;
    ws.onerror = (e) => {
      console.error("[useStreamingTTS] ws error", e);
      toast.error("TTS 连接失败, 请确认 tts_server 已启动 (port 9801)");
      cleanup();
    };
    ws.onclose = () => {
      // 服务端正常关闭 (task 结束后) 或异常关闭
      if (!finishedRef.current) {
        // 异常关闭, 但可能有部分音频已播, 不强制报错
        finishWaiterRef.current?.();
      }
    };

    // 等 ws.onopen
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("tts_server 连接超时")),
        5000,
      );
      ws.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
      ws.onerror = () => {
        clearTimeout(timer);
        reject(new Error("tts_server 连接失败"));
      };
    }).catch((e) => {
      console.error("[useStreamingTTS] start failed", e);
      cleanup();
      return;
    });

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // 发 start
    wsRef.current.send(JSON.stringify({ type: "start" }));

    // 等 ready (DashScope task-started)
    await new Promise<void>((resolve) => {
      const check = () => {
        if (readyRef.current) {
          resolve();
          return;
        }
        setTimeout(check, 20);
      };
      check();
    });
  }, [cleanup, handleWsMessage, setupAudio]);

  /**
   * 喂增量文本. 可多次调.
   * 关键: 如果 task 还没 ready (预热期间 LLM 已经出 token), 缓存到
   * pendingTextRef, ready 回来后 handleWsMessage 会自动 flush.
   * 这样避免 delta 在 ready 之前到达被丢弃 (DashScope 23s 超时 bug 的根因).
   */
  const feedText = useCallback((delta: string) => {
    if (!delta) return;
    fullTextRef.current += delta;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      // ws 还没建好, 缓存 (start 内部 setupAudio 后会建 ws)
      pendingTextRef.current += delta;
      return;
    }
    if (!readyRef.current) {
      // task 还没 ready, 缓存等 ready 后 flush
      pendingTextRef.current += delta;
      return;
    }
    ws.send(JSON.stringify({ type: "text", text: delta }));
  }, []);

  /**
   * 发 finish + 等合成完成.
   * 如果 task 还没 ready, 标记 finishPendingRef, ready 后自动发.
   * 关键: 即使 ws 还没建好 (start 正在连接), 也要标记 pending, 否则
   * LLM 很快结束时 finish 会丢失 → DashScope 23s 超时.
   */
  const finish = useCallback(async () => {
    if (finishedRef.current) return;
    const ws = wsRef.current;

    // ws 还没建好 (start 还在连接): 标记 pending, 等 ready 后发
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      finishPendingRef.current = true;
      console.log("[useStreamingTTS] finish queued (ws not open yet)");
      return;
    }

    if (!readyRef.current) {
      // task 还没 ready, 等 ready 后再发 finish
      finishPendingRef.current = true;
      console.log("[useStreamingTTS] finish queued (waiting for ready)");
    } else {
      // 已 ready, 先 flush 任何 pending text 再发 finish
      if (pendingTextRef.current) {
        ws.send(JSON.stringify({ type: "text", text: pendingTextRef.current }));
        pendingTextRef.current = "";
      }
      ws.send(JSON.stringify({ type: "finish" }));
    }

    // 等 complete 消息 (handleWsMessage 触发)
    if (!finishedRef.current) {
      await new Promise<void>((resolve) => {
        finishWaiterRef.current = resolve;
        // 超时保护: 30s 没完成就放弃等
        setTimeout(() => {
          if (!finishedRef.current) {
            console.warn("[useStreamingTTS] finish timeout");
            resolve();
          }
        }, 30000);
      });
    }
  }, []);

  /** 整段朗读 (手动点喇叭): 检查缓存 → 走完整流程 → 缓存. */
  const speak = useCallback(
    async (text: string, messageId?: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      if (messageId) setSpeakingMessageId(messageId);

      // 1. 缓存命中: 直接播
      const cached = messageCache.get(trimmed);
      if (cached) {
        console.log("[useStreamingTTS] cache HIT, playing cached blob");
        cleanup();
        // 关键: 旧 audio (通常是 MSE 那个) 必须先摘掉所有事件再停,
        // 不要设 src="" — 浏览器会把空 src resolve 成页面 URL, 触发 onerror 误报.
        const old = audioRef.current;
        if (old) {
          old.onended = null;
          old.onerror = null;
          old.onplay = null;
          old.onpause = null;
          old.pause();
        }
        audioRef.current = null;

        const audio = new Audio(cached);
        audioRef.current = audio;
        // cleanup() 会把 speakingMessageId 置 null, 这里在 cleanup 之后重新绑定,
        // 否则喇叭动画不会亮起.
        if (messageId) setSpeakingMessageId(messageId);
        audio.onplay = () => setSpeaking(true);
        audio.onended = () => { setSpeaking(false); setSpeakingMessageId(null); };
        audio.onerror = () => {
          console.warn("[useStreamingTTS] cached audio error");
          setSpeaking(false);
          setSpeakingMessageId(null);
        };
        try {
          await audio.play();
        } catch (e) {
          console.warn("[useStreamingTTS] cached play failed", e);
          setSpeaking(false);
          setSpeakingMessageId(null);
        }
        return;
      }

      // 2. 缓存未命中: 走流式合成
      stop();
      fullTextRef.current = trimmed;
      await start(messageId);
      feedText(trimmed);
      await finish();
    },
    [cleanup, start, feedText, finish],
  );

  /** 立即停止 + 清理. */
  const stop = useCallback(() => {
    const ws = wsRef.current;
    // 发 stop 让后端中止 task
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "stop" }));
      } catch {
        /* noop */
      }
    }
    if (audioRef.current) {
      audioRef.current.pause();
    }
    cleanup();
  }, [cleanup]);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {
          /* noop */
        }
      }
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  return {
    start,
    feedText,
    finish,
    speak,
    stop,
    speaking,
    speakingMessageId,
    loading,
  };
}
