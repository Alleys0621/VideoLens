import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
// logo 已替换为Alleys Sparkles (lucide), 不再使用 LangGraphLogoSVG / GitHubSVG
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  ArrowUp,
  FileX2,
  Globe,
  LoaderCircle,
  Mic,
  PanelRightOpen,
  PanelRightClose,
  Square,
  SquarePen,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { useFileUpload } from "@/hooks/use-file-upload";
import { useStreamingASR } from "@/hooks/useStreamingASR";
import { useAuth } from "@/providers/Auth";
import { useTTSContext } from "@/providers/TTS";
import { getContentString } from "./utils";
import { Volume2 } from "lucide-react";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}


export function Thread({
  videoTimeRef,
  videoControlRef,
}: {
  videoTimeRef?: { current: number };
  videoControlRef?: {
    current: {
      pause: () => void;
      resume: () => void;
      isPaused: () => boolean;
      duckVolume: () => void;
      restoreVolume: () => void;
    } | null;
  };
}) {
  const [threadId, _setThreadId] = useQueryState("threadId");
  // 当前陪看视频目录 (从 URL query 读, page.tsx 的 VideoPlayer 设置)
  const [videoDir] = useQueryState("videoDir", { defaultValue: "" });
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  // 联网模式 toggle: 开启后 refuse 意图 (KB 没相关内容) 时 LLM 自动联网搜索 (DashScope enable_search)
  const [webSearch, setWebSearch] = useState(false);
  // 自动朗读 toggle: AI 回复完成后自动 TTS 播放. 默认开, 持久化到 localStorage.
  const [autoSpeak, setAutoSpeak] = useState(() => {
    if (typeof window === "undefined") return true;
    const stored = window.localStorage.getItem("vl_auto_speak");
    return stored === null ? true : stored === "1";
  });
  const toggleAutoSpeak = () => {
    setAutoSpeak((prev) => {
      const next = !prev;
      if (typeof window !== "undefined") {
        window.localStorage.setItem("vl_auto_speak", next ? "1" : "0");
      }
      return next;
    });
  };
  // 拖拽文件时的视觉反馈 (覆盖层), 替代 toast — 保证两屏都看得见
  const [dragOver, setDragOver] = useState(false);
  // useFileUpload 保留 hook (代码不删), 但 UI 不显示上传控件;
  // 拖拽/粘贴文件时弹 toast 提示不支持, contentBlocks 仅用于 submit 兼容
  const { contentBlocks, setContentBlocks } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;
  // 登录用户的 user_id, 透传给后端 agent (用于 Mem0 记忆按用户隔离)
  const { user } = useAuth();
  // TTS 自动播放: 发送时打"刚发过"标记, 回复完成 (isLoading true→false) 时触发
  // 用 TTSProvider 单例, 跟 ai.tsx 喇叭按钮共享同一个实例
  // (必须在 recorder 之前定义, recorder 的 onPauseOthers 用到它)
  const tts = useTTSContext();

  // 流式 ASR: 边录边识别, 边显示文本
  // 录音开始时停 TTS + 暂停视频 (浏览器 AEC 对媒体回声不可靠)
  // 录音结束时恢复视频
  const recorder = useStreamingASR({
    onPauseOthers: () => {
      tts.stop();
      videoControlRef?.current?.pause();
    },
    onResumeOthers: () => {
      videoControlRef?.current?.resume();
    },
  });

  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const justSentRef = useRef(false);
  const lastSpokenIdRef = useRef<string | null>(null);
  // 已喂给 TTS 的字符位置 (按 message id 重置; 防同一消息被重复喂)
  const lastSpokenLenRef = useRef(0);
  // form ref: ASR 录音停止后自动 submit 用 (避开 setTimeout 后事件 stale 问题)
  const formRef = useRef<HTMLFormElement>(null);

  // TTS 播放时降低视频音量 (2s 淡入淡出), 停止后恢复
  useEffect(() => {
    if (tts.speaking) {
      videoControlRef?.current?.duckVolume();
    } else {
      videoControlRef?.current?.restoreVolume();
    }
  }, [tts.speaking, videoControlRef]);

  const lastError = useRef<string | undefined>(undefined);

  const setThreadId = (id: string | null) => {
    _setThreadId(id);
  };

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  // TODO: this should be part of the useStream hook
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  // ============================================================================
  // 边推边 TTS (MSE 流式): LLM stream 期间, 每个 token 让 messages 变化
  //   → 把新增的 delta 喂给 tts.feedText → ws 推给 tts_server → DashScope
  //   流式合成 → mp3 chunk 流回 → MediaSource 边收边播
  //
  // 预热: handleSubmit 时调 tts.start() (建立 ws + run-task), LLM 出第一个
  //      token 时 ready 已回来, feedText 直接推 DashScope, 首字延迟 ~400ms.
  //
  // 注意: 本 effect 不再 stop + start (会跟 handleSubmit 的 start 冲突 →
  //      "WebSocket closed before connection established"). handleSubmit
  //      已经预热, 这里只负责 feedText + finish.
  //
  // 防误触: 只在 justSentRef=true 时触发 (切 thread / 加载历史不会播)
  // ============================================================================
  useEffect(() => {
    if (!justSentRef.current) return;
    if (!autoSpeak) {
      if (!isLoading) {
        justSentRef.current = false;
        tts.stop();
      }
      return;
    }

    // 找最后一条 AI 消息
    const lastAi = [...messages].reverse().find((m) => m.type === "ai");
    if (!lastAi) return;

    // 新 AI 消息 → 启动 TTS 会话 (唯一入口, 杜绝双重启动)
    if (lastSpokenIdRef.current !== lastAi.id) {
      lastSpokenIdRef.current = lastAi.id ?? null;
      lastSpokenLenRef.current = 0;
      // stop 旧会话 + 启动新会话 (async, 不阻塞)
      tts.stop();
      tts.start(lastAi.id ?? undefined).catch((e) =>
        console.warn("[autoTTS] start failed", e),
      );
    }

    // 喂增量文本 (未 ready 时自动缓存到 pendingTextRef, ready 后 flush)
    const fullText = getContentString(lastAi.content);
    const delta = fullText.slice(lastSpokenLenRef.current);
    lastSpokenLenRef.current = fullText.length;
    if (delta) {
      tts.feedText(delta);
    }

    // LLM stream 结束 → 发 finish, 等合成完成 + 保存缓存
    if (!isLoading) {
      justSentRef.current = false;
      tts.finish().catch((e) => console.warn("[autoTTS] finish failed", e));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, isLoading, autoSpeak]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if ((input.trim().length === 0 && contentBlocks.length === 0) || isLoading)
      return;
    setFirstTokenReceived(false);
    // 标记"刚发过消息" — useEffect 据此启动自动 TTS
    justSentRef.current = true;
    // 停掉之前可能残留的 TTS 会话
    tts.stop();

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [
        ...(input.trim().length > 0 ? [{ type: "text", text: input }] : []),
        ...contentBlocks,
      ] as Message["content"],
    };

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);

    stream.submit(
      { messages: [...toolMessages, newHumanMessage] },
      {
        streamMode: ["values", "messages"],
        streamSubgraphs: true,
        streamResumable: true,
        // video_dir + web_search 通过 config.configurable 传给 graph
        config: {
          configurable: {
            video_dir: videoDir,
            web_search: webSearch,
            video_time: videoTimeRef?.current ?? 0,
            user_id: user?.id ?? "default",
          },
        },
        // thread metadata: 打 user_id 标记, 让 getThreads 能按用户过滤
        // (SDK 在 threadId 为 null 时自动 threads.create({ metadata }) 创建新 thread)
        metadata: { user_id: user?.id ?? "default" },
        optimisticValues: (prev) => ({
          ...prev,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
    setContentBlocks([]);
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
      // 跟 handleSubmit 一样传 configurable, 否则后端拿不到 video_dir / user_id
      config: {
        configurable: {
          video_dir: videoDir,
          web_search: webSearch,
          video_time: videoTimeRef?.current ?? 0,
          user_id: user?.id ?? "default",
        },
      },
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="flex h-full w-full overflow-hidden">
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r bg-white"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div
            className="relative h-full"
            style={{ width: 300 }}
          >
            <ThreadHistory />
          </div>
        </motion.div>
      </div>

      <div className="flex h-full w-full flex-col">
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen
              ? isLargeScreen
                ? "calc(100% - 300px)"
                : "100%"
              : "100%",
          }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          {!chatStarted && (
            <div className="absolute top-0 left-0 z-10 flex w-full items-center justify-between gap-3 p-2 pl-4">
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-gray-100"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? (
                      <PanelRightOpen className="size-5" />
                    ) : (
                      <PanelRightClose className="size-5" />
                    )}
                  </Button>
                )}
              </div>
              <TooltipIconButton
                size="lg"
                className="p-4"
                tooltip="新对话"
                variant="ghost"
                onClick={() => setThreadId(null)}
              >
                <SquarePen className="size-5" />
              </TooltipIconButton>
            </div>
          )}
          {chatStarted && (
            <div className="relative z-10 flex items-center justify-between gap-3 p-2">
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-gray-100"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? (
                      <PanelRightOpen className="size-5" />
                    ) : (
                      <PanelRightClose className="size-5" />
                    )}
                  </Button>
                )}
              </div>

              <div className="flex items-center gap-4">
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="新对话"
                  variant="ghost"
                  onClick={() => setThreadId(null)}
                >
                  <SquarePen className="size-5" />
                </TooltipIconButton>
              </div>

              <div className="from-background to-background/0 absolute inset-x-0 top-full h-5 bg-gradient-to-b" />
            </div>
          )}

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "mt-[25vh] flex flex-col items-stretch",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
              content={
                <>
                  {messages
                    .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message, index) =>
                      message.type === "human" ? (
                        <HumanMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                        />
                      ) : (
                        <AssistantMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                          handleRegenerate={handleRegenerate}
                        />
                      ),
                    )}
                  {/* Special rendering case where there are no AI/tool messages, but there is an interrupt.
                    We need to render it outside of the messages list, since there are no messages to render */}
                  {hasNoAIOrToolMessages && !!stream.interrupt && (
                    <AssistantMessage
                      key="interrupt-msg"
                      message={undefined}
                      isLoading={isLoading}
                      handleRegenerate={handleRegenerate}
                    />
                  )}
                  {isLoading && !firstTokenReceived && (
                    <AssistantMessageLoading />
                  )}
                </>
              }
              footer={
                <div className="sticky bottom-0 flex flex-col items-center gap-8 bg-white">
                  {!chatStarted && (
                    <img
                      src="/alleysvid-logo.png"
                      alt="AlleysVid"
                      className="h-48 w-auto min-[1700px]:h-56"
                    />
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragOver(false);
                      if (e.dataTransfer.files?.length > 0) {
                        toast.info("Alleys暂时收不了文件哦，直接打字跟我聊吧~");
                      }
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(true);
                    }}
                    onDragLeave={(e) => {
                      e.preventDefault();
                      setDragOver(false);
                    }}
                    className="relative z-10 mx-auto mb-8 w-full max-w-3xl rounded-2xl border border-zinc-200/80 bg-white shadow-soft-md transition-all duration-300"
                  >
                    {/* 拖拽文件覆盖层: 两屏尺寸区分 (笔记本/外接), 保证在卡片内可见 */}
                    {dragOver && (
                      <div className="absolute inset-0 z-30 flex items-center justify-center rounded-2xl bg-white/95 backdrop-blur-sm ring-2 ring-dashed ring-zinc-300">
                        <div className="text-center">
                          <FileX2 className="mx-auto h-10 w-10 min-[1700px]:h-12 text-zinc-400" />
                          <p className="mt-3 text-base min-[1700px]:text-lg font-medium text-zinc-600">
                            暂不支持文件上传
                          </p>
                          <p className="mt-1 text-xs min-[1700px]:text-sm text-zinc-400">
                            直接打字跟 Alleys 聊吧~
                          </p>
                        </div>
                      </div>
                    )}
                    <form
                      ref={formRef}
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <textarea
                        value={
                          recorder.isRecording
                            ? (
                                recorder.finalText +
                                " " +
                                recorder.partialText
                              ).trim()
                            : input
                        }
                        readOnly={recorder.isRecording}
                        onChange={(e) => setInput(e.target.value)}
                        onPaste={(e) => {
                          if (e.clipboardData.files?.length > 0) {
                            e.preventDefault();
                            toast.info("Alleys暂时收不了文件哦，直接打字跟我聊吧~");
                          }
                        }}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        enterKeyHint="send"
                        placeholder="给Alleys说点什么..."
                        className="field-sizing-content resize-none border-none bg-transparent px-4 py-3 pb-0 text-[16px] shadow-none ring-0 outline-none placeholder:text-zinc-400 focus:ring-0 focus:outline-none"
                      />

                      <div className="flex items-center justify-between p-2 pt-3">
                        <div className="flex items-center gap-2">
                          {/* 联网模式 toggle: 开启后 KB 检索不相关时 LLM 自动联网 (DashScope enable_search) */}
                          <button
                            type="button"
                            onClick={() => setWebSearch((p) => !p)}
                            className={cn(
                              "flex min-h-[40px] items-center gap-1.5 rounded-full px-4 py-2 text-xs font-medium transition-all",
                              webSearch
                                ? "bg-emerald-500 text-white shadow-sm hover:bg-emerald-600"
                                : "bg-zinc-100 text-zinc-600 ring-1 ring-zinc-200 hover:bg-zinc-200 hover:text-zinc-900",
                            )}
                            title={
                              webSearch
                                ? "联网模式已开启: KB 没相关内容时自动联网搜索"
                                : "联网模式关闭"
                            }
                          >
                            <Globe className="h-3.5 w-3.5" />
                            联网
                          </button>
                          {/* 自动朗读 toggle: AI 回复完成后自动 TTS 播放 */}
                          <button
                            type="button"
                            onClick={toggleAutoSpeak}
                            className={cn(
                              "flex min-h-[40px] items-center gap-1.5 rounded-full px-4 py-2 text-xs font-medium transition-all",
                              autoSpeak
                                ? "bg-indigo-500 text-white shadow-sm hover:bg-indigo-600"
                                : "bg-zinc-100 text-zinc-600 ring-1 ring-zinc-200 hover:bg-zinc-200 hover:text-zinc-900",
                            )}
                            title={
                              autoSpeak
                                ? "自动朗读已开启: AI 回复完成后自动播语音"
                                : "自动朗读关闭 (仍可手动点喇叭按钮)"
                            }
                          >
                            <Volume2 className="h-3.5 w-3.5" />
                            朗读
                          </button>
                        </div>
                        {stream.isLoading ? (
                          <button
                            type="button"
                            onClick={() => stream.stop()}
                            className="flex h-11 items-center gap-1.5 rounded-full bg-zinc-900 px-5 text-sm font-medium text-white shadow-sm transition-all hover:bg-zinc-700"
                          >
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            停止
                          </button>
                        ) : (
                          <button
                            type="button"
                            onClick={async (e) => {
                              if (input.trim()) {
                                (e.currentTarget as HTMLElement)
                                  .closest("form")
                                  ?.requestSubmit();
                              } else if (recorder.isRecording) {
                                // 停止录音: 拿识别结果 → 自动 submit 发给 LLM
                                const text = await recorder.stop();
                                if (text.trim()) {
                                  setInput(text);
                                  // setInput 异步, 等 React re-render 让
                                  // handleSubmit 闭包 capture 到最新 input
                                  setTimeout(() => {
                                    formRef.current?.requestSubmit();
                                  }, 0);
                                } else {
                                  toast.error("没识别到内容, 请重试");
                                }
                              } else {
                                // 开始流式录音 (边录边识别, 边显示)
                                await recorder.start();
                              }
                            }}
                            disabled={
                              recorder.status === "starting" ||
                              recorder.status === "stopping"
                            }
                            className="flex h-11 w-11 items-center justify-center rounded-full bg-zinc-900 text-white shadow-sm transition-all hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {input.trim() && !recorder.isRecording ? (
                              <ArrowUp className="h-4 w-4" />
                            ) : recorder.isRecording ? (
                              <Square className="h-4 w-4" fill="currentColor" />
                            ) : recorder.status === "starting" ||
                              recorder.status === "stopping" ? (
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                            ) : (
                              <Mic className="h-4 w-4" />
                            )}
                          </button>
                        )}
                      </div>
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
      </div>
    </div>
  );
}
