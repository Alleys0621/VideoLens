"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/providers/Auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ChevronDown, KeyRound, LogOut, User as UserIcon } from "lucide-react";
import { toast } from "sonner";

/**
 * 右上角用户菜单: 头像 + 昵称 + 下拉 (修改密码 / 登出).
 * 修改密码走侧边 Sheet (内部表单).
 */
export function UserMenu() {
  const router = useRouter();
  const { user, logout, loading } = useAuth();
  const [open, setOpen] = useState(false);
  const [pwdOpen, setPwdOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // 点外面关闭下拉
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  // 初次加载中也按"未登录"处理, 显示登录按钮 (可点击). 避免之前那种半透明
  // 灰底占位看起来"虚的"、点不开的体验. /me 返回后会自动重渲染.
  if (!user) {
    return (
      <Button
        variant="ghost"
        size="sm"
        className="text-[13px] text-zinc-600"
        disabled={loading}
        onClick={() => router.push("/login")}
      >
        {loading ? "加载中..." : "登录"}
      </Button>
    );
  }

  const initial = user.display_name?.[0]?.toUpperCase() || "U";

  return (
    <>
      <div ref={menuRef} className="relative flex-shrink-0">
        <button
          type="button"
          onClick={() => setOpen((p) => !p)}
          className="flex h-10 items-center gap-1.5 rounded-full bg-white/80 pl-1 pr-2 text-zinc-600 ring-1 ring-zinc-200/80 transition-colors hover:bg-white hover:text-zinc-900"
          title={user.display_name}
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-indigo-400 to-purple-500 text-xs font-semibold text-white">
            {initial}
          </span>
          <span className="hidden max-w-[100px] truncate text-xs sm:inline">
            {user.display_name}
          </span>
          <ChevronDown className="h-3 w-3 text-zinc-400" />
        </button>

        {open && (
          <div className="absolute right-0 top-full z-50 mt-1 w-48 overflow-hidden rounded-lg border border-zinc-200 bg-white py-1 shadow-xl">
            <div className="border-b px-3 py-2">
              <p className="truncate text-sm font-medium text-zinc-800">
                {user.display_name}
              </p>
              <p className="truncate text-[11px] text-zinc-400">
                {user.phone.replace(/(\d{3})\d{4}(\d{4})/, "$1****$2")}
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                setPwdOpen(true);
              }}
              className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-zinc-700 transition-colors hover:bg-zinc-50"
            >
              <KeyRound className="h-4 w-4 text-zinc-400" />
              修改密码
            </button>
            <button
              type="button"
              onClick={async () => {
                setOpen(false);
                try {
                  await logout();
                  toast.success("已登出");
                  router.replace("/login");
                } catch (e) {
                  console.error("[UserMenu] logout failed", e);
                  toast.error("登出失败, 请重试");
                }
              }}
              className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-rose-600 transition-colors hover:bg-rose-50"
            >
              <LogOut className="h-4 w-4" />
              退出登录
            </button>
          </div>
        )}
      </div>

      <ChangePasswordSheet open={pwdOpen} onOpenChange={setPwdOpen} />
    </>
  );
}

function ChangePasswordSheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const [oldP, setOldP] = useState("");
  const [newP, setNewP] = useState("");
  const [confirmP, setConfirmP] = useState("");
  const [loading, setLoading] = useState(false);

  const reset = () => {
    setOldP("");
    setNewP("");
    setConfirmP("");
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    if (newP.length < 6) {
      toast.error("新密码至少 6 位");
      return;
    }
    if (newP !== confirmP) {
      toast.error("两次输入的新密码不一致");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/auth/change-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          old_password: oldP,
          new_password: newP,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        toast.error((data as { error?: string }).error || "修改失败");
      } else {
        toast.success("密码已更新");
        reset();
        onOpenChange(false);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[380px] max-w-[90vw]">
        <SheetHeader>
          <SheetTitle>修改密码</SheetTitle>
          <SheetDescription>
            输入当前密码, 设置新密码 (至少 6 位)
          </SheetDescription>
        </SheetHeader>
        <form
          onSubmit={onSubmit}
          autoComplete="off"
          className="flex flex-1 flex-col gap-4 px-4 py-4"
        >
          <div className="flex flex-col gap-2">
            <Label htmlFor="oldP">当前密码</Label>
            <Input
              id="oldP"
              name="oldP"
              type="password"
              autoComplete="off"
              value={oldP}
              onChange={(e) => setOldP(e.target.value)}
              required
            />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="newP">新密码</Label>
            <Input
              id="newP"
              name="newP"
              type="password"
              autoComplete="new-password"
              value={newP}
              onChange={(e) => setNewP(e.target.value)}
              required
            />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="confirmP">确认新密码</Label>
            <Input
              id="confirmP"
              name="confirmP"
              type="password"
              autoComplete="new-password"
              value={confirmP}
              onChange={(e) => setConfirmP(e.target.value)}
              required
            />
          </div>
          <SheetFooter className="mt-auto">
            <Button
              type="submit"
              disabled={loading}
              className="bg-zinc-900 text-white hover:bg-zinc-700"
            >
              {loading ? "提交中..." : "确认修改"}
            </Button>
          </SheetFooter>
        </form>
      </SheetContent>
    </Sheet>
  );
}
