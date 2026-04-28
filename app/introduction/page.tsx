"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Chart from "chart.js/auto";
import TopNav from "@/components/TopNav";

type VisualState = "damage" | "dirt" | "none" | null;
type InfraredState = "hotspot" | "normal" | null;

export default function IntroductionPage() {
  const [visualState, setVisualState] = useState<VisualState>(null);
  const [infraredState, setInfraredState] = useState<InfraredState>(null);

  const ablationCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const edgeCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const decision = useMemo(() => {
    if (!visualState || !infraredState) {
      return {
        icon: "❓",
        title: "等待完整数据",
        desc: "请同时选择可见光与红外的检测结果。",
        boxClass: "bg-slate-800 border-slate-700",
        iconClass: "text-slate-500"
      };
    }

    if (visualState === "damage" && infraredState === "hotspot") {
      return {
        icon: "⚠",
        title: "确诊：最高级警报",
        desc: "时空强绑定成功！明确存在由物理破损引发的致命热斑病变。",
        boxClass: "bg-rose-900 border-rose-700",
        iconClass: "text-rose-500"
      };
    }

    if (visualState === "dirt" && infraredState === "hotspot") {
      return {
        icon: "✅",
        title: "过滤：假阳性误报",
        desc: "该异常温升系鸟粪/积灰等非结构性污渍遮挡导致，并非不可逆物理损伤。",
        boxClass: "bg-teal-900 border-teal-700",
        iconClass: "text-teal-500"
      };
    }

    if (visualState === "damage" && infraredState === "normal") {
      return {
        icon: "⚠️",
        title: "预警：低危物理潜伏",
        desc: "检测到组件破裂，但尚未引发热斑电阻异常，建议纳入维护计划。",
        boxClass: "bg-amber-900 border-amber-700",
        iconClass: "text-amber-500"
      };
    }

    if (visualState === "dirt" && infraredState === "normal") {
      return {
        icon: "🧹",
        title: "常规：按需清洗标记",
        desc: "检测到表面污点，未引发温升异常，输出至自动化清洗导航库。",
        boxClass: "bg-teal-900 border-teal-700",
        iconClass: "text-teal-500"
      };
    }

    if (visualState === "none" && infraredState === "hotspot") {
      return {
        icon: "🔎",
        title: "异样：内部隐患可能",
        desc: "表面无异常但存在热斑，可能为内部老化或电路脱焊，建议进一步 EL 排查。",
        boxClass: "bg-amber-900 border-amber-700",
        iconClass: "text-amber-500"
      };
    }

    return {
      icon: "✨",
      title: "正常：安全运转",
      desc: "可见光与热力学均未见异常，光伏组件运行状态良好。",
      boxClass: "bg-teal-900 border-teal-700",
      iconClass: "text-teal-500"
    };
  }, [visualState, infraredState]);

  useEffect(() => {
    if (!ablationCanvasRef.current || !edgeCanvasRef.current) return;

    Chart.defaults.font.family = "Segoe UI, Tahoma, Geneva, Verdana, sans-serif";
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.borderColor = "#1e293b";

    const ablation = new Chart(ablationCanvasRef.current, {
      type: "line",
      data: {
        labels: ["YOLO11s Base", "+ CBAM", "+ DySample", "+ NWD Loss", "+ SAHI (Final)"],
        datasets: [
          {
            label: "综合 mAP50 (%)",
            data: [68.5, 70.8, 73.2, 76.5, 83.2],
            backgroundColor: "rgba(37, 99, 235, 0.2)",
            borderColor: "rgba(37, 99, 235, 1)",
            borderWidth: 3,
            pointBackgroundColor: "#fff",
            pointBorderColor: "rgba(37, 99, 235, 1)",
            pointRadius: 5,
            fill: true,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { min: 60, max: 90, ticks: { stepSize: 5 } }
        }
      }
    });

    const edge = new Chart(edgeCanvasRef.current, {
      type: "bar",
      data: {
        labels: ["优化前 (PyTorch原生)", "优化后 (TensorRT量化)"],
        datasets: [
          {
            label: "延迟 Latency (ms)",
            data: [115, 14],
            backgroundColor: "rgba(244, 63, 94, 0.8)",
            borderRadius: 4
          },
          {
            label: "吞吐 FPS",
            data: [8, 55],
            backgroundColor: "rgba(20, 184, 166, 0.8)",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } }
      }
    });

    return () => {
      ablation.destroy();
      edge.destroy();
    };
  }, []);

  const commonBtn =
    "flex-1 rounded border border-slate-600 bg-slate-800 px-3 py-3 text-sm transition hover:bg-slate-700";
  const activeBlue = "bg-blue-600 border-blue-400 text-white";
  const activeRose = "bg-rose-600 border-rose-400 text-white";

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-100 antialiased">
      <div className="mx-auto max-w-7xl px-4 pt-6 sm:px-6 lg:px-8">
        <TopNav active="intro" />
      </div>

      <header className="bg-blue-600 py-20 text-white">
        <div className="mx-auto max-w-7xl px-4 text-center sm:px-6 lg:px-8">
          <h1 className="mb-4 text-4xl font-extrabold tracking-tight md:text-5xl">
            基于改进 YOLO11s 与双光融合
          </h1>
          <h2 className="mb-8 text-2xl font-light md:text-3xl">无人机光伏边缘智能巡检系统</h2>
          <p className="mx-auto mb-10 max-w-3xl text-lg text-blue-100">
            打破传统光伏电站人工巡检效率低下与航拍漏检痛点。将轻量化视觉模型部署于边缘设备，首创“三重保险”双光因果互证机制，彻底终结复杂户外环境下的虚假报警。
          </p>
          <div className="mx-auto grid max-w-4xl grid-cols-2 gap-4 md:grid-cols-4">
            <div className="rounded-lg bg-blue-700/50 p-4 backdrop-blur-sm">
              <div className="text-3xl font-bold">83.2%</div>
              <div className="mt-1 text-sm text-blue-200">综合 mAP50 精度</div>
            </div>
            <div className="rounded-lg bg-blue-700/50 p-4 backdrop-blur-sm">
              <div className="text-3xl font-bold">
                14<span className="text-xl">ms</span>
              </div>
              <div className="mt-1 text-sm text-blue-200">单切片推理延迟</div>
            </div>
            <div className="rounded-lg bg-blue-700/50 p-4 backdrop-blur-sm">
              <div className="text-3xl font-bold">
                14.2<span className="text-xl">MB</span>
              </div>
              <div className="mt-1 text-sm text-blue-200">量化后引擎体积</div>
            </div>
            <div className="rounded-lg bg-blue-700/50 p-4 backdrop-blur-sm">
              <div className="text-3xl font-bold">三重</div>
              <div className="mt-1 text-sm text-blue-200">去伪存真保险机制</div>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-24 px-4 py-12 sm:px-6 lg:px-8">
        <section id="overview" className="scroll-mt-20">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-bold text-slate-100">行业痛点与系统概览</h2>
            <div className="mx-auto mt-4 h-1 w-20 rounded bg-blue-600" />
          </div>
          <p className="mx-auto mb-8 max-w-4xl text-center text-slate-300">
            本部分介绍了光伏智能运维面临的两大核心挑战，以及本系统如何通过“端侧感知、底层魔改、边缘加速、逻辑闭环”四大步骤重构巡检范式。系统的核心价值在于精准捕获微小缺陷并坚决排除环境干扰。
          </p>
          <div className="grid gap-8 md:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-8 shadow-sm">
              <h3 className="mb-4 text-xl font-bold text-rose-600">⚠ 现有技术的致命瓶颈</h3>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="mr-2 text-rose-500">✖</span>
                  <div>
                    <strong className="block text-slate-800">超高分辨率下的“特征淹没”</strong>
                    <span className="text-sm text-slate-600">
                      4K航拍视场宏大，微小裂纹/缺角像素占比极小，常规下采样极易丢失高频特征导致漏检。
                    </span>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-rose-500">✖</span>
                  <div>
                    <strong className="block text-slate-800">复杂户外背景下的“高误报率”</strong>
                    <span className="text-sm text-slate-600">
                      玻璃反光、鸟粪、局部积灰等非结构性污渍极易被单一可见光模型误判为物理病变。
                    </span>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-rose-500">✖</span>
                  <div>
                    <strong className="block text-slate-800">云端依赖导致的时效性缺失</strong>
                    <span className="text-sm text-slate-600">
                      海量高清图像回传极度依赖上行带宽，无法满足偏远电站的实时预警需求。
                    </span>
                  </div>
                </li>
              </ul>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-8 shadow-sm">
              <h3 className="mb-4 text-xl font-bold text-teal-600">✔ 本项目的破局之道</h3>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="mr-2 text-teal-500">➤</span>
                  <div>
                    <strong className="block text-slate-800">底层架构魔改：极小目标无损捕捉</strong>
                    <span className="text-sm text-slate-600">
                      引入 CBAM 注意力排杂、DySample 动态放大特征、NWD Loss 克服梯度弥散，结合 SAHI 切片推理。
                    </span>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-teal-500">➤</span>
                  <div>
                    <strong className="block text-slate-800">多模态融合：双光因果互证</strong>
                    <span className="text-sm text-slate-600">
                      将可见光结构损伤与红外异常温升进行空间强绑定，彻底过滤伪缺陷假阳性。
                    </span>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-teal-500">➤</span>
                  <div>
                    <strong className="block text-slate-800">极致边缘部署：断网级实时计算</strong>
                    <span className="text-sm text-slate-600">
                      基于 TensorRT 混合精度量化，引擎压缩至 14MB，Jetson 节点实现 55+ FPS 吞吐量。
                    </span>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </section>

        <section id="algorithm" className="scroll-mt-20">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-bold text-slate-100">面向微小缺陷的底层网络魔改</h2>
            <div className="mx-auto mt-4 h-1 w-20 rounded bg-blue-600" />
          </div>
          <p className="mx-auto mb-8 max-w-4xl text-center text-lg text-slate-300">
            本区域展示了系统核心算法的底层创新设计。为了解决微小目标特征易平滑、易丢失的问题，我们对 YOLO11s 的特征聚合与损失计算层面进行了三项关键性的前沿技术替换，并通过 HTML 结构图展示了特征处理的流水线逻辑。
          </p>
          <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-8 shadow-sm">
            <h3 className="mb-8 text-center text-lg font-bold text-slate-200">改进特征提取流水线</h3>
            <div className="flex flex-col items-center justify-center gap-4 text-center md:flex-row md:gap-8">
              <div className="w-full rounded-lg border-2 border-slate-700 bg-slate-800 p-6 md:w-48">
                <div className="text-2xl">📷</div>
                <div className="font-bold text-slate-100">输入层</div>
                <div className="mt-2 text-xs text-slate-400">SAHI 自适应重叠切片 4K 高清图像</div>
              </div>
              <div className="text-2xl text-slate-400">→</div>
              <div className="w-full rounded-lg border-2 border-blue-800/60 bg-blue-900/30 p-6 md:w-48">
                <div className="text-2xl">🔍</div>
                <div className="font-bold text-blue-800">CBAM</div>
                <div className="mt-2 text-xs text-blue-600">
                  通道与空间双重聚焦，强力抑制背景噪点反光
                </div>
              </div>
              <div className="text-2xl text-slate-400">→</div>
              <div className="w-full rounded-lg border-2 border-teal-800/60 bg-teal-900/30 p-6 md:w-48">
                <div className="text-2xl">📈</div>
                <div className="font-bold text-teal-800">DySample</div>
                <div className="mt-2 text-xs text-teal-600">
                  动态采样点生成，非均匀放大微小拓扑高频特征
                </div>
              </div>
              <div className="text-2xl text-slate-400">→</div>
              <div className="w-full rounded-lg border-2 border-purple-800/60 bg-purple-900/30 p-6 md:w-48">
                <div className="text-2xl">🎯</div>
                <div className="font-bold text-purple-800">NWD Loss</div>
                <div className="mt-2 text-xs text-purple-600">
                  二维高斯分布建模，彻底克服极小目标梯度弥散
                </div>
              </div>
            </div>
            <div className="mt-10 grid gap-6 md:grid-cols-3">
              <div className="rounded border border-slate-800 bg-slate-800/70 p-4">
                <strong className="mb-2 block text-slate-100">排杂与提纯 (CBAM)</strong>
                <p className="text-sm text-slate-300">
                  赋予模型类似人类视觉的筛选能力，自适应拉高物理损伤权重，引导算力锚定视场中极微小的缺陷区域，保障底层特征学习纯粹性。
                </p>
              </div>
              <div className="rounded border border-slate-800 bg-slate-800/70 p-4">
                <strong className="mb-2 block text-slate-100">动态特征放大 (DySample)</strong>
                <p className="text-sm text-slate-300">
                  摒弃固定插值规则，通过生成器网络学习采样点偏移量，有效捕获那些原本在传统上采样中会被过度平滑掉的关键尖角与边缘。
                </p>
              </div>
              <div className="rounded border border-slate-800 bg-slate-800/70 p-4">
                <strong className="mb-2 block text-slate-100">解决梯度弥散 (NWD Loss)</strong>
                <p className="text-sm text-slate-300">
                  用 Wasserstein 距离替代传统 IoU。即使极小预测框与真实框完全不交叠，依然能提供连续有效梯度，确保定位稳定性。
                </p>
              </div>
            </div>
          </div>
        </section>

        <section id="logic" className="scroll-mt-20">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-bold text-slate-100">第三重保险：双光因果互证决策实验室</h2>
            <div className="mx-auto mt-4 h-1 w-20 rounded bg-blue-600" />
          </div>
          <p className="mx-auto mb-8 max-w-4xl text-center text-lg text-slate-300">
            本部分为交互式演示区。系统的灵魂在于彻底消除户外复杂环境导致的误报。请手动组合“可见光视觉”与“红外热成像”的检测结果，体验系统如何基于“物理损伤引发温升”的因果强绑定逻辑，输出精准诊断结论。
          </p>
          <div className="rounded-xl border border-slate-700 bg-slate-900 p-6 text-white shadow-lg md:p-10">
            <div className="mb-8 grid gap-8 md:grid-cols-2">
              <div className="space-y-6">
                <div>
                  <h4 className="mb-3 text-lg font-medium text-slate-300">
                    步骤 1: 可见光网络检测结果 (模拟)
                  </h4>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setVisualState("damage")}
                      className={`${commonBtn} ${visualState === "damage" ? activeBlue : ""}`}
                    >
                      结构性物理损伤
                      <br />
                      <span className="text-xs text-slate-300">(隐裂/缺角/碎裂)</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setVisualState("dirt")}
                      className={`${commonBtn} ${visualState === "dirt" ? activeBlue : ""}`}
                    >
                      非结构性表面污渍
                      <br />
                      <span className="text-xs text-slate-300">(鸟粪/局部积灰)</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setVisualState("none")}
                      className={`${commonBtn} ${visualState === "none" ? activeBlue : ""}`}
                    >
                      面板表面清洁
                      <br />
                      <span className="text-xs text-slate-300">(无可见异常)</span>
                    </button>
                  </div>
                </div>
                <div>
                  <h4 className="mb-3 text-lg font-medium text-slate-300">
                    步骤 2: 红外热成像网络检测结果 (模拟)
                  </h4>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setInfraredState("hotspot")}
                      className={`${commonBtn} ${infraredState === "hotspot" ? activeRose : ""}`}
                    >
                      探测到异常高温区域
                      <br />
                      <span className="text-xs text-slate-300">(显著热斑)</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setInfraredState("normal")}
                      className={`${commonBtn} ${infraredState === "normal" ? activeRose : ""}`}
                    >
                      温度分布均匀
                      <br />
                      <span className="text-xs text-slate-300">(无热力异常)</span>
                    </button>
                  </div>
                </div>
              </div>
              <div
                className={`flex flex-col items-center justify-center rounded-lg border p-6 text-center ${decision.boxClass}`}
              >
                <h4 className="mb-4 text-sm font-bold uppercase tracking-widest text-slate-300">
                  系统智能确诊输出
                </h4>
                <div className={`mb-4 text-6xl ${decision.iconClass}`}>{decision.icon}</div>
                <div className="mb-2 text-2xl font-bold">{decision.title}</div>
                <div className="max-w-xs text-sm text-slate-200">{decision.desc}</div>
              </div>
            </div>
            <div className="rounded border-l-4 border-blue-500 bg-slate-800/50 p-4 text-sm text-slate-300">
              <strong>理论基础：</strong>
              系统内嵌了严格的物理因果律——仅当“物理破损引起局部电阻增大”进而导致“热斑病变”在同一空间坐标下重叠触发时，才确诊为高危病灶。这从算法根源上排除了环境干扰引发的无效报警。
            </div>
          </div>
        </section>

        <section id="performance" className="scroll-mt-20">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-bold text-slate-100">消融实验与边缘部署效能</h2>
            <div className="mx-auto mt-4 h-1 w-20 rounded bg-blue-600" />
          </div>
          <p className="mx-auto mb-8 max-w-4xl text-center text-lg text-slate-300">
            本部分以量化数据证明改进方案的卓越性。图表展示了各底层模块叠加对综合检测精度（mAP50）的提升轨迹，以及模型在 Jetson 边缘节点量化部署后的吞吐能力与轻量化优势。
          </p>
          <div className="mb-8 grid gap-8 md:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-6 shadow-sm">
              <h3 className="mb-2 text-center text-lg font-bold text-slate-100">
                消融实验：mAP50 精度提升轨迹
              </h3>
              <p className="mb-6 text-center text-xs text-slate-400">验证底层魔改与推理插件对精度的绝对贡献</p>
              <div className="relative mx-auto h-[40vh] max-h-[400px] w-full max-w-[800px] md:h-[350px]">
                <canvas ref={ablationCanvasRef} />
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-6 shadow-sm">
              <h3 className="mb-2 text-center text-lg font-bold text-slate-100">
                边缘部署：TensorRT 加速效能对比
              </h3>
              <p className="mb-6 text-center text-xs text-slate-400">
                突破断网级大带宽限制，实现实时推理计算
              </p>
              <div className="relative mx-auto h-[40vh] max-h-[400px] w-full max-w-[800px] md:h-[350px]">
                <canvas ref={edgeCanvasRef} />
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-cyan-900/40 bg-cyan-900/10 p-6 text-slate-200">
            <h4 className="mb-4 text-lg font-bold text-cyan-300">数据核心结论解析</h4>
            <ul className="list-disc space-y-2 pl-5 text-sm">
              <li>
                <strong>精度飞跃：</strong> 最终方案（融合 CBAM, DySample, NWD, 配合 SAHI 推理）相较于 YOLO11s 基线模型，综合 mAP50 从 68.5% 跃升至{" "}
                <strong>83.2%</strong>，实现了 14.7% 的绝对精度跨越。
              </li>
              <li>
                <strong>计算极速：</strong> 经过 FP16/INT8 量化，边缘端单次切片前向计算延迟骤降至{" "}
                <strong>14 毫秒</strong>，切片吞吐算力飙升至 <strong>55+ FPS</strong>，配合 SAHI 可实现 4K 全图 ~2.4 FPS 的综合处理速度。
              </li>
              <li>
                <strong>极致瘦身：</strong> 最终部署引擎体积仅 <strong>14.2MB</strong>，边缘节点功耗控制在 15W，打通复杂视觉模型在低功耗硬件落地的最后一公里。
              </li>
            </ul>
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-800 bg-slate-900 py-8 text-slate-400">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 sm:px-6 lg:flex-row lg:px-8">
          <div>
            <span className="font-bold tracking-wider text-white">PV-INSPECT AI</span>
            <span className="ml-2 text-sm">智能巡检系统数据可视化面板</span>
          </div>
          <div className="text-sm">基于多模态因果互证机制的绿色智慧运维新范式</div>
        </div>
      </footer>
    </div>
  );
}
