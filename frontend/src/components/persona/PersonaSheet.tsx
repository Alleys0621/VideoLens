"use client";

import { useEffect, useState } from "react";
import { useQueryState } from "nuqs";
import { toast } from "sonner";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { PERSONAS, getPersona } from "@/lib/personas";
import { useThreads } from "@/providers/Thread";
import { useAuth } from "@/providers/Auth";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

export function PersonaSheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const { user } = useAuth();
  const { threads, updateThreadMetadata, getThreads } = useThreads();
  const [threadId] = useQueryState("threadId");
  const [defaultPersonaId, setDefaultPersonaId] = useState("alleys");
  const [saving, setSaving] = useState(false);

  const threadMeta = threads.find((t) => t.thread_id === threadId)
    ?.metadata as Record<string, unknown> | null;
  const threadPersonaId = (threadMeta?.persona_id as string | undefined) || defaultPersonaId;
  const threadPersona = getPersona(threadPersonaId);

  useEffect(() => {
    if (!open || !user) return;
    fetch("/api/preferences/persona", { cache: "no-store" })
      .then((r) => r.json())
      .then((data: { persona_id?: string }) => {
        if (data.persona_id) setDefaultPersonaId(data.persona_id);
      })
      .catch(() => undefined);
  }, [open, user]);

  const saveDefault = async (id: string) => {
    if (saving) return;
    setSaving(true);
    try {
      const res = await fetch("/api/preferences/persona", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona_id: id }),
      });
      const data = (await res.json().catch(() => ({}))) as { error?: string };
      if (!res.ok) throw new Error(data.error || "保存失败");
      setDefaultPersonaId(id);
      toast.success(`默认搭子已切换为${getPersona(id).name}`);
    } catch (e) {
      toast.error((e as Error).message || "保存失败");
    } finally {
      setSaving(false);
    }
  };

  const saveThread = async (id: string) => {
    if (!threadId || saving) return;
    setSaving(true);
    try {
      await updateThreadMetadata(threadId, { persona_id: id });
      await getThreads();
      toast.success(`当前对话已切换为${getPersona(id).name}，之后的回复由她来`);
    } catch (e) {
      toast.error((e as Error).message || "切换失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[420px] max-w-[92vw] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>搭子</SheetTitle>
          <SheetDescription>选择陪你追剧的人设</SheetDescription>
        </SheetHeader>
        <div className="flex flex-col gap-6 px-4 py-4">
          <section>
            <h3 className="text-sm font-semibold text-zinc-700">默认搭子</h3>
            <p className="mb-3 mt-1 text-xs text-zinc-400">新建对话时默认使用</p>
            <PersonaGrid selected={defaultPersonaId} onSelect={saveDefault} />
          </section>
          {threadId && user && (
            <section>
              <h3 className="text-sm font-semibold text-zinc-700">当前对话</h3>
              <p className="mb-3 mt-1 text-xs text-zinc-400">
                只影响这个对话，之后的新回复由 {threadPersona.name} 来
              </p>
              <PersonaGrid selected={threadPersonaId} onSelect={saveThread} />
            </section>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function PersonaGrid({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-3">
      {PERSONAS.map((p) => {
        const active = p.id === selected;
        return (
          <button
            key={p.id}
            type="button"
            onClick={() => onSelect(p.id)}
            className={cn(
              "flex items-center gap-3 rounded-lg border p-3 text-left transition-colors",
              active
                ? "border-zinc-900 bg-zinc-50 ring-1 ring-zinc-900"
                : "border-zinc-200 hover:bg-zinc-50",
            )}
          >
            <img src={p.avatar} alt={p.name} className="h-11 w-11 rounded-lg object-cover" />
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-semibold text-zinc-800">{p.name}</span>
              <span className="block text-xs text-zinc-400">{p.tagline}</span>
            </span>
            {active && <Check className="h-4 w-4 text-zinc-900" />}
          </button>
        );
      })}
    </div>
  );
}
