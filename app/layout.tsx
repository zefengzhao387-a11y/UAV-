import "./globals.css";
import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "UAV 光伏缺陷智能检测端侧系统",
  description: "基于 Next.js 与远端 ONNX 推理服务的无人机光伏缺陷智能检测演示",
  icons: {
    icon: "/favicon.svg"
  }
};

interface RootLayoutProps {
  children: ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
