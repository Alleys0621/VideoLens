"use client";

import { useState, FormEvent } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/providers/Auth";
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
import { MessageCircle, Smartphone, QrCode } from "lucide-react";
import { toast } from "sonner";

const PHONE_RE = /^1[3-9]\d{9}$/;

export default function LoginPage() {
  const router = useRouter();
  const search = useSearchParams();
  const from = search.get("from") || "/";
  const { login } = useAuth();

  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  // 手机号失焦检查结果: idle | checking | not_exists | ok | invalid
  const [phoneCheck, setPhoneCheck] = useState<
    "idle" | "checking" | "not_exists" | "ok" | "invalid"
  >("idle");

  // 失焦检查手机号是否已注册
  const checkPhone = async () => {
    const p = phone.trim();
    if (!p) {
      setPhoneCheck("idle");
      return;
    }
    if (!PHONE_RE.test(p)) {
      setPhoneCheck("invalid");
      return;
    }
    setPhoneCheck("checking");
    try {
      const res = await fetch(
        `/api/auth/check-phone?phone=${encodeURIComponent(p)}`,
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setPhoneCheck("idle");
        return;
      }
      setPhoneCheck((data as { exists: boolean }).exists ? "ok" : "not_exists");
    } catch {
      setPhoneCheck("idle");
    }
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (loading) return;
    if (phoneCheck === "not_exists") {
      toast.error("该手机号未注册");
      return;
    }
    setLoading(true);
    try {
      await login(phone.trim(), password);
      toast.success("登录成功");
      router.replace(from);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "登录失败");
    } finally {
      setLoading(false);
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
          <CardTitle className="mt-2 text-xl">欢迎回来</CardTitle>
          <CardDescription>登录以继续和 Alleys 一起看剧</CardDescription>
        </CardHeader>

        <form onSubmit={onSubmit} autoComplete="off">
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="phone">手机号</Label>
              <Input
                id="phone"
                name="phone"
                type="tel"
                inputMode="numeric"
                pattern="[0-9]*"
                maxLength={11}
                autoComplete="off"
                placeholder="请输入手机号 (仅数字)"
                value={phone}
                onChange={(e) => {
                  setPhone(e.target.value.replace(/\D/g, "").slice(0, 11));
                  setPhoneCheck("idle");
                }}
                onBlur={checkPhone}
                required
              />
              {phoneCheck === "invalid" && (
                <p className="text-xs text-rose-500">手机号格式不正确</p>
              )}
              {phoneCheck === "checking" && (
                <p className="text-xs text-zinc-400">检查中...</p>
              )}
              {phoneCheck === "not_exists" && (
                <p className="text-xs text-rose-500">
                  该手机号未注册,{" "}
                  <Link
                    href="/register"
                    className="font-medium text-indigo-500 hover:text-indigo-600"
                  >
                    去注册
                  </Link>
                </p>
              )}
              {phoneCheck === "ok" && (
                <p className="text-xs text-emerald-500">该手机号已注册</p>
              )}
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">密码</Label>
                <Link
                  href="/forgot-password"
                  className="text-xs text-indigo-500 hover:text-indigo-600"
                >
                  忘记密码?
                </Link>
              </div>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete="off"
                placeholder="请输入密码"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                disabled={phoneCheck === "not_exists"}
              />
            </div>
          </CardContent>

          <CardFooter className="mt-2 flex flex-col gap-3">
            <Button
              type="submit"
              disabled={loading || phoneCheck === "not_exists"}
              className="h-10 w-full bg-zinc-900 text-white hover:bg-zinc-700"
            >
              {loading ? "登录中..." : "登录"}
            </Button>

            <div className="flex w-full items-center gap-3 py-1 text-xs text-zinc-400">
              <div className="h-px flex-1 bg-zinc-200" />
              其他登录方式
              <div className="h-px flex-1 bg-zinc-200" />
            </div>

            <div className="flex w-full gap-2">
              <Button
                type="button"
                variant="outline"
                disabled
                title="短信通道开通后可用"
                className="h-10 flex-1 cursor-not-allowed opacity-60"
              >
                <Smartphone className="h-4 w-4" />
                验证码登录
              </Button>
              <Button
                type="button"
                variant="outline"
                disabled
                title="微信扫码登录开通后可用"
                className="h-10 flex-1 cursor-not-allowed opacity-60"
              >
                <QrCode className="h-4 w-4" />
                微信扫码
              </Button>
            </div>

            <p className="text-center text-xs text-zinc-500">
              还没有账号?{" "}
              <Link
                href="/register"
                className="font-medium text-indigo-500 hover:text-indigo-600"
              >
                立即注册
              </Link>
            </p>
          </CardFooter>
        </form>

        <div className="mt-2 flex items-center justify-center gap-1.5 text-[11px] text-zinc-400">
          <MessageCircle className="h-3 w-3" />
          <span>Alleys — AI 陪看智能体</span>
        </div>
      </Card>
    </div>
  );
}
