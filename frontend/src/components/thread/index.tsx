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
import { useAudioRecorder } from "@/hooks/useAudioRecorder";

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


export function Thread() {
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
  // 拖拽文件时的视觉反馈 (覆盖层), 替代 toast — 保证两屏都看得见
  const [dragOver, setDragOver] = useState(false);
  // useFileUpload 保留 hook (代码不删), 但 UI 不显示上传控件;
  // 拖拽/粘贴文件时弹 toast 提示不支持, contentBlocks 仅用于 submit 兼容
  const { contentBlocks, setContentBlocks } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const recorder = useAudioRecorder();
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

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

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if ((input.trim().length === 0 && contentBlocks.length === 0) || isLoading)
      return;
    setFirstTokenReceived(false);

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
          },
        },
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
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <textarea
                        value={input}
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
                        placeholder="给Alleys说点什么..."
                        className="field-sizing-content resize-none border-none bg-transparent px-4 py-3 pb-0 text-[15px] shadow-none ring-0 outline-none placeholder:text-zinc-400 focus:ring-0 focus:outline-none"
                      />

                      <div className="flex items-center justify-between p-2 pt-3">
                        {/* 联网模式 toggle: 开启后 KB 检索不相关时 LLM 自动联网 (DashScope enable_search) */}
                        <button
                          type="button"
                          onClick={() => setWebSearch((p) => !p)}
                          className={cn(
                            "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all",
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
                        {stream.isLoading ? (
                          <button
                            type="button"
                            onClick={() => stream.stop()}
                            className="flex h-9 items-center gap-1.5 rounded-full bg-zinc-900 px-4 text-sm font-medium text-white shadow-sm transition-all hover:bg-zinc-700"
                          >
                            <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
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
                                const blob = await recorder.stopRecording();
                                if (blob) {
                                  const text = await recorder.transcribe(blob);
                                  if (text) setInput(text);
                                }
                              } else {
                                await recorder.startRecording();
                              }
                            }}
                            disabled={recorder.isTranscribing}
                            className="flex h-9 w-9 items-center justify-center rounded-full bg-zinc-900 text-white shadow-sm transition-all hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {input.trim() && !recorder.isRecording ? (
                              <ArrowUp className="h-4 w-4" />
                            ) : recorder.isRecording ? (
                              <Square className="h-4 w-4" fill="currentColor" />
                            ) : recorder.isTranscribing ? (
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
