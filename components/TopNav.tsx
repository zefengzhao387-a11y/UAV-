"use client";

import Link from "next/link";

interface TopNavProps {
  active: "detect" | "intro";
}

export default function TopNav({ active }: TopNavProps) {
  const tabBase =
    "rounded-md px-3 py-1 text-xs md:text-sm transition-all duration-300 border";
  const activeTab = "border-cyan-300 bg-cyan-500/20 text-cyan-200 shadow-sm";
  const idleTab = "border-cyan-400/30 text-cyan-300 hover:border-cyan-300 hover:text-cyan-200";

  return (
    <header className="mb-6">
      <div className="rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-slate-900 to-slate-950 px-5 py-4 shadow-lg shadow-cyan-500/5">
        <div className="flex items-center justify-between gap-3">
          <h1 className="truncate text-lg font-semibold tracking-wide text-cyan-300 md:text-xl">
            UAV 光伏缺陷智能检测端侧系统
          </h1>
          <div className="flex items-center gap-2">
            <Link href="/" className={`${tabBase} ${active === "detect" ? activeTab : idleTab}`}>
              检测系统
            </Link>
            <Link
              href="/introduction"
              className={`${tabBase} ${active === "intro" ? activeTab : idleTab}`}
            >
              项目简介
            </Link>
            <span className="hidden text-xs text-slate-400 md:block">ONNX Runtime Web · Edge AI</span>
          </div>
        </div>
      </div>
    </header>
  );
}
