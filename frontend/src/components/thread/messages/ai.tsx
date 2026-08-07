import { parsePartialJson } from "@langchain/core/output_parsers";
import { useStreamContext } from "@/providers/Stream";
import { AIMessage, Checkpoint, Message } from "@langchain/langgraph-sdk";
import { getContentString } from "../utils";
import { CommandBar } from "./shared";
import { MarkdownText } from "../markdown-text";
import { cn } from "@/lib/utils";
import { ToolCalls, ToolResult } from "./tool-calls";
import { MessageContentComplex } from "@langchain/core/messages";
import { useQueryState, parseAsBoolean } from "nuqs";
import { ReasoningCard } from "../ReasoningCard";
import { TooltipIconButton } from "../tooltip-icon-button";
import { Volume2, LoaderCircle, Square } from "lucide-react";
import { useTTSContext } from "@/providers/TTS";
import { getPersona } from "@/lib/personas";

function parseAnthropicStreamedToolCalls(
  content: MessageContentComplex[],
): AIMessage["tool_calls"] {
  const toolCallContents = content.filter((c) => c.type === "tool_use" && c.id);

  return toolCallContents.map((tc) => {
    const toolCall = tc as Record<string, any>;
    let json: Record<string, any> = {};
    if (toolCall?.input) {
      try {
        json = parsePartialJson(toolCall.input) ?? {};
      } catch {
        // Pass
      }
    }
    return {
      name: toolCall.name ?? "",
      id: toolCall.id ?? "",
      args: json,
      type: "tool_call",
    };
  });
}

export function AssistantMessage({
  message,
  isLoading,
  handleRegenerate,
}: {
  message: Message | undefined;
  isLoading: boolean;
  handleRegenerate: (parentCheckpoint: Checkpoint | null | undefined) => void;
}) {
  const content = message?.content ?? [];
  const contentString = getContentString(content);
  const [hideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );

  const thread = useStreamContext();
  const meta = message ? thread.getMessagesMetadata(message) : undefined;
  const tts = useTTSContext();
  const persona = getPersona(
    (message as Record<string, any>)?.additional_kwargs?.persona_id as
      | string
      | undefined,
  );

  const parentCheckpoint = meta?.firstSeenState?.parent_checkpoint;
  const anthropicStreamedToolCalls = Array.isArray(content)
    ? parseAnthropicStreamedToolCalls(content)
    : undefined;

  const hasToolCalls =
    message &&
    "tool_calls" in message &&
    message.tool_calls &&
    message.tool_calls.length > 0;
  const toolCallsHaveContents =
    hasToolCalls &&
    message.tool_calls?.some(
      (tc) => tc.args && Object.keys(tc.args).length > 0,
    );
  const hasAnthropicToolCalls = !!anthropicStreamedToolCalls?.length;
  const isToolResult = message?.type === "tool";

  if (isToolResult && hideToolCalls) {
    return null;
  }

  return (
    <div className="group mr-auto flex w-full items-start gap-3">
      {/* 按 persona_id 显示对应头像 */}
      <img
        src={persona.avatar}
        alt={persona.name}
        className="h-8 w-8 flex-shrink-0 rounded-lg object-cover shadow-sm"
      />
      <div className="flex flex-1 flex-col gap-2">
        {isToolResult ? (
          <ToolResult message={message} />
        ) : (
          <>
            {contentString.length > 0 && (
              <div className="mb-1 text-xs font-medium text-zinc-400">
                {persona.name}
              </div>
            )}
            {contentString.length > 0 && (
              /* ChatGPT 风格: 无气泡背景, Markdown 直接渲染 */
              <div className="text-[15px] leading-relaxed text-zinc-800">
                <MarkdownText>{contentString}</MarkdownText>
              </div>
            )}

            {/* 推理卡片 (GPT 思考过程风格): 只在检索类意图显示 (kb/kb_meta/web_search),
                闲聊/拒答不显示 — 避免无检索时还冒出推理框 */}
            {message &&
              (() => {
                const r = (message as Record<string, any>)?.additional_kwargs
                  ?.reasoning;
                if (
                  !r ||
                  !["kb", "kb_meta", "web_search"].includes(r.intent)
                )
                  return null;
                return <ReasoningCard reasoning={r} />;
              })()}

            {!hideToolCalls && (
              <>
                {(hasToolCalls && toolCallsHaveContents && (
                  <ToolCalls toolCalls={message.tool_calls} />
                )) ||
                  (hasAnthropicToolCalls && (
                    <ToolCalls toolCalls={anthropicStreamedToolCalls} />
                  )) ||
                  (hasToolCalls && (
                    <ToolCalls toolCalls={message.tool_calls} />
                  ))}
              </>
            )}

            <div
              className={cn(
                "flex items-center gap-2",
              )}
            >
              <CommandBar
                content={contentString}
                isLoading={isLoading}
                isAiMessage={true}
                handleRegenerate={() => handleRegenerate(parentCheckpoint)}
              />
              {/* TTS 朗读按钮: 仅当前正在播放的消息显示动态喇叭 */}
              {contentString.length > 0 && (() => {
                const isThisSpeaking = tts.speaking && tts.speakingMessageId === message?.id;
                return (
                  <TooltipIconButton
                    tooltip={isThisSpeaking ? "停止朗读" : "朗读"}
                    variant="ghost"
                    onClick={() =>
                      isThisSpeaking ? tts.stop() : tts.speak(contentString, message?.id)
                    }
                  >
                    {isThisSpeaking ? (
                      <Volume2 className="h-4 w-4 animate-tts-speaking text-indigo-500" />
                    ) : tts.loading && tts.speakingMessageId === message?.id ? (
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                    ) : (
                      <Volume2 className="h-4 w-4" />
                    )}
                  </TooltipIconButton>
                );
              })()}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function AssistantMessageLoading() {
  return (
    <div className="mr-auto flex w-full items-start gap-3">
      <img
        src="/alleysvid-avatar.png"
        alt="Alleys"
        className="h-8 w-8 flex-shrink-0 rounded-lg object-cover shadow-sm"
      />
      <div className="flex-1 pt-2">
        <div className="flex items-center gap-1.5">
          <div className="h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_infinite] rounded-full bg-zinc-400/60"></div>
          <div className="h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_0.5s_infinite] rounded-full bg-zinc-400/60"></div>
          <div className="h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_1s_infinite] rounded-full bg-zinc-400/60"></div>
        </div>
      </div>
    </div>
  );
}
