"use client";

import { useState } from "react";
import { Brain, Globe } from "lucide-react";

type RetrievedEvent = { event_id?: string; title?: string; score?: number };
type WebResult = { title?: string; url?: string; content?: string; score?: number };
type Reasoning = {
  intent?: string;
  query?: string;
  top_score?: number;
  threshold?: number;
  retrieved?: RetrievedEvent[];
  selected?: { event_id?: string; title?: string }[];
  web_results?: WebResult[];
};

const INTENT_LABEL: Record<string, string> = {
  kb: "检索了知识库",
  kb_meta: "基于剧情概要",
  web_search: "联网搜索了",
};

/**
 * 推理过程卡片 (GPT 思考过程风格).
 * 折叠态: 极简灰字 + 图标, 无背景无边框, 不抢主回答焦点.
 *   - kb/kb_meta → Brain 图标
 *   - web_search → Globe 图标 (联网)
 * 展开态: 淡灰虚化背景 + 斜体小字, 和正常回答有明显视觉区分.
 */
export function ReasoningCard({ reasoning }: { reasoning: Reasoning | null }) {
  const [open, setOpen] = useState(false);
  if (!reasoning || !reasoning.intent) return null;

  const label = INTENT_LABEL[reasoning.intent] ?? "思考过程";
  const Icon = reasoning.intent === "web_search" ? Globe : Brain;
  const hasWebResults =
    reasoning.intent === "web_search" &&
    reasoning.web_results &&
    reasoning.web_results.length > 0;

  return (
    <div className="my-1.5">
      <button
        onClick={() => setOpen((p) => !p)}
        className="flex items-center gap-1.5 py-0.5 text-[11px] text-zinc-400 transition-colors hover:text-zinc-600"
      >
        <Icon className="h-3 w-3" />
        <span>{label}</span>
        {reasoning.intent === "kb" && reasoning.top_score !== undefined && (
          <span className="text-zinc-300">· 相关度 {reasoning.top_score}</span>
        )}
        {hasWebResults && (
          <span className="text-zinc-300">
            · {reasoning.web_results!.length} 条结果
          </span>
        )}
        <span className="ml-0.5 text-zinc-300">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="mt-1 rounded-md bg-zinc-50/80 px-3 py-2 text-[11px] leading-relaxed text-zinc-500 backdrop-blur-[1px]">
          {hasWebResults ? (
            <div className="space-y-1.5 italic">
              <div className="not-italic text-zinc-400">
                搜索结果 (top {reasoning.web_results!.length}):
              </div>
              {reasoning.web_results!.map((r, i) => (
                <div key={i} className="flex flex-col gap-0.5">
                  <a
                    href={r.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="not-italic truncate text-blue-500 hover:underline"
                  >
                    {i + 1}. {r.title || r.url}
                  </a>
                  {r.url && (
                    <span className="truncate text-[10px] text-zinc-400">
                      {r.url}
                    </span>
                  )}
                  {r.content && (
                    <span className="line-clamp-2 text-zinc-500">
                      {r.content}
                    </span>
                  )}
                </div>
              ))}
            </div>
          ) : reasoning.retrieved && reasoning.retrieved.length > 0 ? (
            <div className="space-y-0.5 italic">
              <div className="not-italic text-zinc-400">
                检索到的事件 (top{" "}
                {Math.min(5, reasoning.retrieved.length)}):
              </div>
              {reasoning.retrieved.slice(0, 5).map((e, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="h-1 w-1 flex-shrink-0 rounded-full bg-zinc-300" />
                  <span className="flex-1 truncate">
                    {e.title || e.event_id}
                  </span>
                  {e.score !== undefined && (
                    <span className="flex-shrink-0 font-mono text-[10px] text-zinc-400">
                      {e.score}
                    </span>
                  )}
                </div>
              ))}
              {reasoning.selected && reasoning.selected.length > 0 && (
                <div className="pt-1 text-zinc-400">
                  → 选中 {reasoning.selected.length} 个作为回答依据
                </div>
              )}
            </div>
          ) : reasoning.intent === "web_search" ? (
            <div className="italic text-zinc-400">
              联网搜索了但未返回结果
            </div>
          ) : (
            <div className="italic text-zinc-400">
              基于视频概要回答, 未走事件检索
            </div>
          )}
        </div>
      )}
    </div>
  );
}
