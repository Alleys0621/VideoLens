import type { Metadata } from "next";
import "./globals.css";
import { Inter, JetBrains_Mono, Exo_2 } from "next/font/google";
import React from "react";
import { NuqsAdapter } from "nuqs/adapters/next/app";

// Inter: 拉丁正文; JetBrains Mono: 视频状态栏/等宽数字
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});
// Exo 2: 流线科技感, 用于 AlleysVid 品牌字 (蓝橙渐变文字, 匹配 logo 配色)
const exo2 = Exo_2({
  subsets: ["latin"],
  weight: "700",
  variable: "--font-exo2",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Alleys — AI 陪看智能体",
  description: "AI 驱动的视频陪聊智能体系统，边看边聊剧情",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className={`${inter.variable} ${jetbrainsMono.variable} ${exo2.variable}`}>
        <NuqsAdapter>{children}</NuqsAdapter>
      </body>
    </html>
  );
}
