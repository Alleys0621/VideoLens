import { useStreamContext } from "@/providers/Stream";
import { Message } from "@langchain/langgraph-sdk";
import { getContentString } from "../utils";
import { cn } from "@/lib/utils";
import { CommandBar } from "./shared";
import { MultimodalPreview } from "@/components/thread/MultimodalPreview";
import { isBase64ContentBlock } from "@/lib/multimodal-utils";

export function HumanMessage({
  message,
  isLoading,
}: {
  message: Message;
  isLoading: boolean;
}) {
  const thread = useStreamContext();
  const contentString = getContentString(message.content);

  return (
    <div className="group ml-auto flex w-full items-center gap-3">
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex min-w-0 flex-col gap-2">
          {/* 已上传文件预览 (base64 content blocks, 保留文件上传能力) */}
          {Array.isArray(message.content) && message.content.length > 0 && (
            <div className="flex flex-wrap items-end justify-end gap-2">
              {message.content.reduce<React.ReactNode[]>((acc, block, idx) => {
                if (isBase64ContentBlock(block)) {
                  acc.push(
                    <MultimodalPreview key={idx} block={block} size="md" />,
                  );
                }
                return acc;
              }, [])}
            </div>
          )}
          {/* ChatGPT 风格用户气泡: 浅灰底, 大圆角, 右对齐, 无头像 */}
          {contentString ? (
            <p className="ml-auto w-fit max-w-[85%] min-w-0 whitespace-pre-wrap [overflow-wrap:break-word] rounded-3xl bg-[#f4f4f4] px-4 py-2.5 text-[15px] leading-relaxed text-zinc-900">
              {contentString}
            </p>
          ) : null}
        </div>

        <div
          className={cn(
            "ml-auto flex items-center gap-2 transition-opacity",
            "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100",
          )}
        >
          <CommandBar isLoading={isLoading} content={contentString} />
        </div>
      </div>
    </div>
  );
}
