"use client";

import { useState, FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
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
import { toast } from "sonner";

const PHONE_RE = /^1[3-9]\d{9}$/;

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [confirmP, setConfirmP] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [loading, setLoading] = useState(false);
  // 手机号失焦检查结果: idle | checking | exists | ok | invalid
  const [phoneCheck, setPhoneCheck] = useState<
    "idle" | "checking" | "exists" | "ok" | "invalid"
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
      setPhoneCheck((data as { exists: boolean }).exists ? "exists" : "ok");
    } catch {
      setPhoneCheck("idle");
    }
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (loading) return;

    if (!PHONE_RE.test(phone.trim())) {
      toast.error("手机号格式不正确");
      return;
    }
    if (password.length < 6) {
      toast.error("密码至少 6 位");
      return;
    }
    if (password !== confirmP) {
      toast.error("两次输入的密码不一致");
      return;
    }
    if (phoneCheck === "exists") {
      toast.error("该手机号已注册");
      return;
    }

    setLoading(true);
    try {
      await register({
        phone: phone.trim(),
        password,
        display_name: displayName.trim() || undefined,
      });
      toast.success("注册成功, 请登录");
      // 注册和登录分离: 跳到登录页
      router.replace("/login");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "注册失败");
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
          <CardTitle className="mt-2 text-xl">创建账号</CardTitle>
          <CardDescription>用手机号注册, 即可开始陪看</CardDescription>
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
                placeholder="11 位手机号 (仅数字)"
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
              {phoneCheck === "exists" && (
                <p className="text-xs text-rose-500">
                  该手机号已注册,{" "}
                  <Link
                    href="/login"
                    className="font-medium text-indigo-500 hover:text-indigo-600"
                  >
                    去登录
                  </Link>
                </p>
              )}
              {phoneCheck === "ok" && (
                <p className="text-xs text-emerald-500">该手机号可以注册</p>
              )}
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="display_name">昵称 (可选)</Label>
              <Input
                id="display_name"
                name="display_name"
                type="text"
                autoComplete="off"
                placeholder="不填会自动生成"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                maxLength={20}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="password">密码</Label>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete="new-password"
                placeholder="至少 6 位"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="confirm">确认密码</Label>
              <Input
                id="confirm"
                name="confirm"
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
              disabled={loading || phoneCheck === "exists"}
              className="h-10 w-full bg-zinc-900 text-white hover:bg-zinc-700"
            >
              {loading ? "注册中..." : "注册"}
            </Button>

            <p className="text-center text-xs text-zinc-500">
              已有账号?{" "}
              <Link
                href="/login"
                className="font-medium text-indigo-500 hover:text-indigo-600"
              >
                返回登录
              </Link>
            </p>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
