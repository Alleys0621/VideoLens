"use client";

import { useStreamContext } from "@/providers/Stream";

export type KeyframeMeta = {
  timestamp: number;
  path: string;
};

type AdditionalKwargs = {
  keyframes?: KeyframeMeta[];
  reasoning?: { intent?: string; [k: string]: unknown } | null;
};

type MessageWithAdditional = {
  additional_kwargs?: AdditionalKwargs;
  type?: string;
  [k: string]: unknown;
};

/**
 * 从最新 AI message 的 additional_kwargs.keyframes / reasoning 提取数据,
 * 供 VideoPlayer 的推理信息条展示"定位"按钮和推理卡片.
 *
 * 不再自动 seek —— 用户主动点关键帧按钮才跳转 (避免 Alleys 回复带 keyframe 时
 * 把用户正在看的画面切走). 跳转逻辑在 VideoPlayer 的按钮 onClick 里:
 *   onClick={() => { if (artRef.current) artRef.current.currentTime = kf.timestamp; }}
 */
export function useKeyframeSeek() {
  const stream = useStreamContext();
  const messages = stream.messages;

  const lastMsg = messages?.[messages.length - 1] as
    | MessageWithAdditional
    | undefined;
  const keyframes: KeyframeMeta[] = lastMsg?.additional_kwargs?.keyframes ?? [];
  const reasoning = lastMsg?.additional_kwargs?.reasoning ?? null;
  return { keyframes, reasoning };
}
