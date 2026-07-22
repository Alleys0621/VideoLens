"use client";

import { useState, FormEvent } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { toast } from "sonner";
import { ArrowLeft } from "lucide-react";

const PHONE_RE = /^1[3-9]\d{9}$/;

export default function ForgotPasswordPage() {
  const [phone, setPhone] = useState("");
  const [code, setCode] = useState("");
  const [newP, setNewP] = useState("");
  const [confirmP, setConfirmP] = useState("");
  const [sending, setSending] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const onSendCode = async () => {
    if (!PHONE_RE.test(phone.trim())) {
      toast.error("手机号格式不正确");
      return;
    }
    setSending(true);
    try {
      const res = await fetch("/api/auth/sms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone: phone.trim(), scene: "reset" }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        toast.error(
          (data as { error?: string }).error || "验证码发送失败",
        );
      } else {
        toast.success("验证码已发送");
      }
    } finally {
      setSending(false);
    }
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    if (newP.length < 6) {
      toast.error("新密码至少 6 位");
      return;
    }
    if (newP !== confirmP) {
      toast.error("两次输入的密码不一致");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          phone: phone.trim(),
          code,
          new_password: newP,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        toast.error(
          (data as { error?: string }).error || "重置失败",
        );
      } else {
        toast.success("密码已重置, 请返回登录");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#F0F2F5] p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="items-center text-center">
          <img
            src="/alleysvid-logo.png"
            alt="AlleysVid"
            className="mx-auto h-14 w-auto"
          />
          <CardTitle className="mt-2 text-xl">找回密码</CardTitle>
          <CardDescription>
            通过手机验证码重置密码
            <br />
            <span className="text-amber-600">
              (当前短信通道未开通, 提交会提示失败)
            </span>
          </CardDescription>
        </CardHeader>

        <form onSubmit={onSubmit} autoComplete="off">
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="phone">手机号</Label>
              <div className="flex gap-2">
                <Input
                  id="phone"
                  name="phone"
                  type="tel"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={11}
                  autoComplete="off"
                  placeholder="11 位手机号 (仅数字)"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value.replace(/\D/g, "").slice(0, 11))}
                  required
                />
                <Button
                  type="button"
                  variant="outline"
                  disabled={sending}
                  onClick={onSendCode}
                  className="h-9 flex-shrink-0"
                >
                  {sending ? "发送中..." : "发送验证码"}
                </Button>
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="code">验证码</Label>
              <Input
                id="code"
                name="code"
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                autoComplete="one-time-code"
                placeholder="6 位数字"
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                maxLength={6}
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
                placeholder="至少 6 位"
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
                placeholder="再输入一次"
                value={confirmP}
                onChange={(e) => setConfirmP(e.target.value)}
                required
              />
            </div>
          </CardContent>

          <CardFooter className="mt-2 flex flex-col gap-3">
            <Button
              type="submit"
              disabled={submitting}
              className="h-10 w-full bg-zinc-900 text-white hover:bg-zinc-700"
            >
              {submitting ? "提交中..." : "重置密码"}
            </Button>

            <Link
              href="/login"
              className="flex items-center justify-center gap-1 text-xs text-zinc-500 hover:text-zinc-700"
            >
              <ArrowLeft className="h-3 w-3" />
              返回登录
            </Link>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
