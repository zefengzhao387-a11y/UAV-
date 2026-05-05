"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ImageUploadCard from "@/components/ImageUploadCard";
import TopNav from "@/components/TopNav";
import type { DetectionBox, PostprocessDebugInfo } from "@/utils/modelHelper";
import { drawBoundingBoxes } from "@/utils/modelHelper";
import {
  inferServiceConfigured,
  runYoloInferRemote,
  useRemoteInfer
} from "@/utils/inferApi";
import { runYoloInferClient } from "@/utils/yoloClientInfer";

const VISIBLE_CLASSES = ["Clean", "Dust", "Bird", "Electrical", "Physical", "Snow"];
// 官方微调后导出的热力模型为单类 hotspot。
const THERMAL_CLASSES = ["Hotspot"];
/** 产品上热力框统一使用该展示名（含 nc=2 时解析出的 Class-1 等）。 */
const HOTSPOT_DISPLAY_NAME = "Hotspot";
const STRUCTURAL_DAMAGE_CLASSES = new Set([
  "physical",
  "electrical",
  "crack",
  "brokencell",
  "fragment",
  "star_crack"
]);
const CROSS_MODAL_IOU_THRESHOLD = 0.06;

/** 演示模式：将同名 PNG 放到 public/demo/ 下即可替换前后对比图（无需改代码）。 */
const DEMO_VISIBLE_IN = "/demo/demo-visible-in.png";
const DEMO_THERMAL_IN = "/demo/demo-thermal-in.png";
const DEMO_VISIBLE_OUT = "/demo/demo-visible-out.png";
const DEMO_THERMAL_OUT = "/demo/demo-thermal-out.png";

const DEPLOYED_INFER_ORIGIN = "https://uav.onrender.com";

type DemoImagePair = { visible: string; thermal: string };

interface FusionResult {
  highRiskCount: number;
  structuralCandidateCount: number;
  hotspotCandidateCount: number;
}

function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return url;
}

export default function AnalysisWorkspace() {
  const remoteInfer = useRemoteInfer();
  const [visibleFile, setVisibleFile] = useState<File | null>(null);
  const [thermalFile, setThermalFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const inferConfigured = useMemo(() => inferServiceConfigured(), []);
  const [statusText, setStatusText] = useState(() =>
    !useRemoteInfer()
      ? "等待上传两张图像（推理在浏览器内执行，首次会下载 ONNX 与 WASM）"
      : inferServiceConfigured()
        ? "等待上传两张图像（推理在服务端执行）"
        : "请在环境变量 NEXT_PUBLIC_INFER_SERVICE_URL 中配置后端推理服务的完整 origin"
  );
  const [errorText, setErrorText] = useState<string | null>(null);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [fusionResult, setFusionResult] = useState<FusionResult>({
    highRiskCount: 0,
    structuralCandidateCount: 0,
    hotspotCandidateCount: 0
  });
  const [visibleDebug, setVisibleDebug] = useState<PostprocessDebugInfo | null>(null);
  const [thermalDebug, setThermalDebug] = useState<PostprocessDebugInfo | null>(null);
  const [visibleScoreThreshold, setVisibleScoreThreshold] = useState(0.52);
  const [visibleIouThreshold, setVisibleIouThreshold] = useState(0.55);
  // 热力分支按 Ultralytics 常规检测参数
  const [thermalScoreThreshold, setThermalScoreThreshold] = useState(0.25);
  const [thermalIouThreshold, setThermalIouThreshold] = useState(0.45);
  const [demoUrls, setDemoUrls] = useState<DemoImagePair | null>(null);
  const [isDemoRunning, setIsDemoRunning] = useState(false);

  const visibleUrl = useObjectUrl(visibleFile);
  const thermalUrl = useObjectUrl(thermalFile);
  const effectiveVisibleUrl = demoUrls?.visible ?? visibleUrl;
  const effectiveThermalUrl = demoUrls?.thermal ?? thermalUrl;

  const visibleImgRef = useRef<HTMLImageElement | null>(null);
  const thermalImgRef = useRef<HTMLImageElement | null>(null);
  const visibleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const thermalCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const canAnalyze = useMemo(
    () =>
      Boolean(visibleUrl && thermalUrl && visibleFile && thermalFile && inferConfigured) &&
      !isAnalyzing,
    [visibleUrl, thermalUrl, visibleFile, thermalFile, inferConfigured, isAnalyzing]
  );

  const clearCanvas = (canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const normalizeBox = (
    box: DetectionBox,
    width: number,
    height: number
  ): DetectionBox => ({
    ...box,
    xMin: box.xMin / width,
    yMin: box.yMin / height,
    xMax: box.xMax / width,
    yMax: box.yMax / height
  });

  const calcIou = (a: DetectionBox, b: DetectionBox): number => {
    const x1 = Math.max(a.xMin, b.xMin);
    const y1 = Math.max(a.yMin, b.yMin);
    const x2 = Math.min(a.xMax, b.xMax);
    const y2 = Math.min(a.yMax, b.yMax);
    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const interArea = interW * interH;
    if (interArea <= 0) return 0;
    const areaA = Math.max(0, a.xMax - a.xMin) * Math.max(0, a.yMax - a.yMin);
    const areaB = Math.max(0, b.xMax - b.xMin) * Math.max(0, b.yMax - b.yMin);
    const union = areaA + areaB - interArea;
    return union > 0 ? interArea / union : 0;
  };

  const visibleAnalyzeOptions = {
    classNames: VISIBLE_CLASSES,
    hasObjectness: false as const,
    minBoxSize: 2,
    maxBoxAreaRatio: 0.8,
    scoreThreshold: visibleScoreThreshold,
    iouThreshold: visibleIouThreshold,
    maxDetections: 120,
    boxFormat: "cxcywh" as const
  };

  const thermalAnalyzeOptions = {
    classNames: THERMAL_CLASSES,
    hasObjectness: false as const,
    applySigmoid: false,
    fallbackToAlternateHead: false,
    minBoxSize: 0,
    maxBoxAreaRatio: 1,
    scoreThreshold: thermalScoreThreshold,
    iouThreshold: thermalIouThreshold,
    maxDetections: 100,
    boxFormat: "cxcywh" as const
  };

  const waitImageReady = async (
    imageRef: React.RefObject<HTMLImageElement | null>
  ): Promise<HTMLImageElement> => {
    const element = imageRef.current;
    if (!element) {
      throw new Error("图片元素不存在，请重新上传后再试");
    }

    if (element.complete && element.naturalWidth > 0) {
      return element;
    }

    return new Promise((resolve, reject) => {
      const handleLoad = () => resolve(element);
      const handleError = () => reject(new Error("图片加载失败，无法执行推理"));
      element.addEventListener("load", handleLoad, { once: true });
      element.addEventListener("error", handleError, { once: true });
    });
  };

  const handleVisibleSelected = (file: File) => {
    setDemoUrls(null);
    setVisibleFile(file);
  };

  const handleThermalSelected = (file: File) => {
    setDemoUrls(null);
    setThermalFile(file);
  };

  const runDemo = async () => {
    if (isAnalyzing || isDemoRunning) return;
    setIsDemoRunning(true);
    setErrorText(null);
    setVisibleDebug(null);
    setThermalDebug(null);
    setFusionResult({
      highRiskCount: 0,
      structuralCandidateCount: 0,
      hotspotCandidateCount: 0
    });
    setVisibleFile(null);
    setThermalFile(null);
    clearCanvas(visibleCanvasRef.current);
    clearCanvas(thermalCanvasRef.current);

    setDemoUrls({ visible: DEMO_VISIBLE_IN, thermal: DEMO_THERMAL_IN });
    setStatusText("演示模式：已载入示例输入图（无真实模型推理）…");

    const step = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
    try {
      await step(700);
      setStatusText("正在分析可见光图像…");
      await step(2200);
      setStatusText("正在分析红外热力图…");
      await step(2200);

      setDemoUrls({ visible: DEMO_VISIBLE_OUT, thermal: DEMO_THERMAL_OUT });
      setFusionResult({
        highRiskCount: 1,
        structuralCandidateCount: 12,
        hotspotCandidateCount: 14
      });
      setStatusText("");
    } finally {
      setIsDemoRunning(false);
    }
  };

  const runAnalyze = async () => {
    if (!canAnalyze) return;
    setErrorText(null);
    setFusionResult({ highRiskCount: 0, structuralCandidateCount: 0, hotspotCandidateCount: 0 });
    setVisibleDebug(null);
    setThermalDebug(null);
    setIsAnalyzing(true);

    try {
      const inferExec = remoteInfer ? runYoloInferRemote : runYoloInferClient;

      const [visibleImage, thermalImage] = await Promise.all([
        waitImageReady(visibleImgRef),
        waitImageReady(thermalImgRef)
      ]);

      if (!visibleFile || !thermalFile) {
        throw new Error("缺少原始图像文件，请重新选择图片");
      }

      setStatusText("正在分析可见光图像...");
      let visibleBoxes: DetectionBox[];
      if (debugEnabled) {
        const result = await inferExec(
          visibleFile,
          "visible",
          visibleAnalyzeOptions,
          true,
          setStatusText
        );
        visibleBoxes = result.boxes;
        setVisibleDebug(result.debug ?? null);
      } else {
        const result = await inferExec(
          visibleFile,
          "visible",
          visibleAnalyzeOptions,
          false,
          setStatusText
        );
        visibleBoxes = result.boxes;
      }

      // 只展示“结构性物理损伤”关键类别，减少画面噪声，评委更容易理解闭环。
      const structuralVisibleBoxes = visibleBoxes.filter((box) =>
        STRUCTURAL_DAMAGE_CLASSES.has(box.className.toLowerCase())
      );
      drawBoundingBoxes(
        visibleCanvasRef.current!,
        visibleImage,
        structuralVisibleBoxes
      );

      setStatusText("正在分析红外热力图...");
      let thermalBoxes: DetectionBox[];
      if (debugEnabled) {
        const result = await inferExec(
          thermalFile,
          "thermal",
          thermalAnalyzeOptions,
          true,
          setStatusText
        );
        thermalBoxes = result.boxes;
        setThermalDebug(result.debug ?? null);
      } else {
        const result = await inferExec(
          thermalFile,
          "thermal",
          thermalAnalyzeOptions,
          false,
          setStatusText
        );
        thermalBoxes = result.boxes;
      }

      // 多类 head（如 nc=2）时仍可能解析出 Class-1：统一改成 Hotspot 以便绘制与闭环统计一致。
      thermalBoxes = thermalBoxes.map((box) => ({
        ...box,
        className: HOTSPOT_DISPLAY_NAME,
        classId: 0
      }));

      const hotspotBoxes = thermalBoxes;

      drawBoundingBoxes(thermalCanvasRef.current!, thermalImage, thermalBoxes);

      // 双光因果闭环：
      // 仅当“结构性物理损伤”与“热斑”在归一化二维坐标上重叠触发，才确诊高危病灶。
      const visibleNorm = structuralVisibleBoxes.map((box) =>
        normalizeBox(box, visibleImage.naturalWidth, visibleImage.naturalHeight)
      );
      const thermalNorm = hotspotBoxes.map((box) =>
        normalizeBox(box, thermalImage.naturalWidth, thermalImage.naturalHeight)
      );

      let highRiskCount = 0;
      for (const vBox of visibleNorm) {
        const hasOverlap = thermalNorm.some(
          (tBox) => calcIou(vBox, tBox) >= CROSS_MODAL_IOU_THRESHOLD
        );
        if (hasOverlap) highRiskCount += 1;
      }

      setFusionResult({
        highRiskCount,
        structuralCandidateCount: structuralVisibleBoxes.length,
        hotspotCandidateCount: hotspotBoxes.length
      });

      if (highRiskCount > 0) {
        setStatusText(
          `高危告警：发现 ${highRiskCount} 处“结构损伤+热斑”同位重叠病灶（已通过因果闭环确认）`
        );
      } else {
        setStatusText(
          `闭环过滤完成：可见光结构损伤 ${structuralVisibleBoxes.length} 处，红外热斑 ${hotspotBoxes.length} 处，但未形成同位重叠高危病灶`
        );
      }
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "分析失败");
      setStatusText(
        remoteInfer ? "分析失败，请检查推理服务、网络与图像" : "分析失败：请确认 public/model/ 中 ONNX 可访问（或 NEXT_PUBLIC_*_MODEL_URL），并查看控制台"
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    clearCanvas(visibleCanvasRef.current);
  }, [visibleUrl, demoUrls?.visible]);

  useEffect(() => {
    clearCanvas(thermalCanvasRef.current);
  }, [thermalUrl, demoUrls?.thermal]);

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-4 pb-10 pt-6 md:px-8">
      <TopNav active="detect" />
      <div className="mb-4 mt-2 text-center">
        <h1 className="text-2xl font-bold tracking-wide text-cyan-200 md:text-3xl">
          缺陷检测模型演示
        </h1>
      </div>

      <section className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <ImageUploadCard
          title="可见光图像 (Visible Light Image)"
          hint="用于识别裂纹、碎片、污染与遮挡等缺陷"
          imageUrl={effectiveVisibleUrl}
          onFileSelected={handleVisibleSelected}
          imageRef={visibleImgRef}
          canvasRef={visibleCanvasRef}
        />

        <ImageUploadCard
          title="红外热力图 (Thermal Image)"
          hint="用于识别热点与热异常区域"
          imageUrl={effectiveThermalUrl}
          onFileSelected={handleThermalSelected}
          imageRef={thermalImgRef}
          canvasRef={thermalCanvasRef}
        />
      </section>

      <section className="mt-7 flex flex-col items-center gap-3">
        {!remoteInfer ? (
          <p className="w-full max-w-3xl text-center text-xs text-slate-400">
            推理链路：
            <span className="text-slate-300">服务端 Ultralytics/YOLO 推理 · 远端 API</span>
            <span className="text-slate-500"> · </span>
            <span className="text-slate-400">模型部署</span>{" "}
            <a
              href={DEPLOYED_INFER_ORIGIN}
              className="text-cyan-300 underline decoration-cyan-500/40 underline-offset-2 hover:text-cyan-200"
              target="_blank"
              rel="noopener noreferrer"
            >
              {DEPLOYED_INFER_ORIGIN}
            </a>
          </p>
        ) : !inferConfigured ? (
          <div className="panel w-full max-w-3xl border border-rose-900/70 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">
            <p className="font-semibold text-rose-100">推理服务地址未配置</p>
            <p className="mt-2 text-xs text-rose-200/90">
              远程模式需在 Vercel / 本地环境中添加{" "}
              <code className="rounded bg-slate-950 px-1 py-0.5 font-mono text-slate-200">
                NEXT_PUBLIC_INFER_SERVICE_URL
              </code>{" "}
              （例如{" "}
              <code className="rounded bg-slate-950 px-1 py-0.5 font-mono">{DEPLOYED_INFER_ORIGIN}</code>
              ，无尾部斜杠）。默认由本站{" "}
              <code className="font-mono">/api/infer-proxy</code> 转发。
            </p>
            <p className="mt-2 text-xs text-slate-300">
              若改回仅在浏览器推理，移除{" "}
              <code className="font-mono">NEXT_PUBLIC_USE_REMOTE_INFER</code>。
            </p>
          </div>
        ) : (
          <p className="w-full max-w-3xl text-center text-xs text-slate-400">
            推理链路：
            <span className="text-slate-300">服务端 Ultralytics/YOLO 推理 · 远端 API</span>
            <span className="text-slate-500"> · </span>
            <span className="text-slate-400">模型部署</span>{" "}
            <a
              href={DEPLOYED_INFER_ORIGIN}
              className="text-cyan-300 underline decoration-cyan-500/40 underline-offset-2 hover:text-cyan-200"
              target="_blank"
              rel="noopener noreferrer"
            >
              {DEPLOYED_INFER_ORIGIN}
            </a>
            <span className="text-slate-500">（首次推理可能较慢）</span>
          </p>
        )}
        <div className="panel w-full max-w-3xl px-4 py-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="text-sm text-slate-200">
              <span className="font-semibold text-cyan-200">Debug</span> 解析面板
              <div className="mt-1 text-xs text-slate-400">
                显示模型输出维度、obj/坐标格式选择与 Top 框置信度（用于定位“乱框”原因）
              </div>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-200">
              <input
                type="checkbox"
                checked={debugEnabled}
                disabled={isAnalyzing}
                onChange={(e) => setDebugEnabled(e.target.checked)}
              />
              开启
            </label>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <label className="flex flex-col gap-2 text-sm text-slate-200">
              <span>
                可见光 Score:{" "}
                <span className="font-semibold text-cyan-300">
                  {visibleScoreThreshold.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min={0.1}
                max={0.95}
                step={0.01}
                value={visibleScoreThreshold}
                disabled={isAnalyzing}
                onChange={(event) => setVisibleScoreThreshold(Number(event.target.value))}
                className="accent-cyan-400"
              />
            </label>

            <label className="flex flex-col gap-2 text-sm text-slate-200">
              <span>
                可见光 IoU:{" "}
                <span className="font-semibold text-cyan-300">
                  {visibleIouThreshold.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.01}
                value={visibleIouThreshold}
                disabled={isAnalyzing}
                onChange={(event) => setVisibleIouThreshold(Number(event.target.value))}
                className="accent-cyan-400"
              />
            </label>

            <label className="flex flex-col gap-2 text-sm text-slate-200">
              <span>
                红外 Score:{" "}
                <span className="font-semibold text-cyan-300">
                  {thermalScoreThreshold.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min={0.1}
                max={0.95}
                step={0.01}
                value={thermalScoreThreshold}
                disabled={isAnalyzing}
                onChange={(event) => setThermalScoreThreshold(Number(event.target.value))}
                className="accent-cyan-400"
              />
            </label>

            <label className="flex flex-col gap-2 text-sm text-slate-200">
              <span>
                红外 IoU:{" "}
                <span className="font-semibold text-cyan-300">
                  {thermalIouThreshold.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.01}
                value={thermalIouThreshold}
                disabled={isAnalyzing}
                onChange={(event) => setThermalIouThreshold(Number(event.target.value))}
                className="accent-cyan-400"
              />
            </label>
          </div>
          <p className="mt-3 text-xs text-slate-400">
            调优建议：可见光微裂缝可把 Score 调到 0.35~0.55；红外热点可把 Score 调到 0.6~0.8。
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            disabled={!canAnalyze}
            onClick={runAnalyze}
            className="rounded-lg bg-cyan-500 px-8 py-3 font-medium text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
          >
            {isAnalyzing ? "分析中..." : "开始缺陷分析"}
          </button>
          <button
            type="button"
            disabled={isAnalyzing || isDemoRunning}
            onClick={runDemo}
            className="rounded-lg border border-cyan-500/60 bg-slate-900 px-6 py-3 font-medium text-cyan-200 transition hover:border-cyan-400 hover:text-cyan-100 disabled:cursor-not-allowed disabled:border-slate-600 disabled:text-slate-500"
          >
            {isDemoRunning ? "演示进行中…" : "开始演示"}
          </button>
        </div>
        {statusText ? <p className="text-sm text-slate-300">{statusText}</p> : null}
        <div className="panel w-full max-w-3xl px-4 py-3 text-sm">
          <p className="text-slate-200">
            可见光结构损伤候选：<span className="text-cyan-300">{fusionResult.structuralCandidateCount}</span>
            {" · "}
            红外热斑候选：<span className="text-cyan-300">{fusionResult.hotspotCandidateCount}</span>
            {" · "}
            同位闭环高危病灶：<span className="text-rose-400">{fusionResult.highRiskCount}</span>
          </p>
          <p className="mt-1 text-xs text-slate-400">
            判定规则：仅当可见光“结构性物理损伤”与红外“热斑”在同一二维坐标系下重叠触发时，输出最高级别报警。
          </p>
        </div>

        {debugEnabled && (visibleDebug || thermalDebug) && (
          <div className="panel w-full max-w-3xl px-4 py-4 text-sm">
            <div className="mb-3 font-semibold text-cyan-200">Debug 输出解析</div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
                <div className="mb-2 text-xs font-semibold text-slate-300">可见光（Visible）</div>
                {!visibleDebug ? (
                  <div className="text-xs text-slate-400">等待...</div>
                ) : (
                  <div className="space-y-1 text-xs text-slate-300">
                    <div>output dims: {visibleDebug.outputDims.join("x")}</div>
                    <div>hasObjectness: {String(visibleDebug.effectiveHasObjectness)}</div>
                    <div>boxFormat: {visibleDebug.effectiveBoxFormat}</div>
                    <div>
                      classes: start={visibleDebug.classStart}, num={visibleDebug.numClasses}
                    </div>
                    <div>returned boxes: {visibleDebug.returnedBoxes}</div>
                    <div>
                      Top5:
                      <div className="mt-1 space-y-0.5">
                        {visibleDebug.topBoxes.length === 0
                          ? "无"
                          : visibleDebug.topBoxes.map((b, idx) => (
                              <div key={idx}>
                                {b.className} {Math.round(b.score * 100)}%
                              </div>
                            ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
                <div className="mb-2 text-xs font-semibold text-slate-300">红外热力（Thermal）</div>
                {!thermalDebug ? (
                  <div className="text-xs text-slate-400">等待...</div>
                ) : (
                  <div className="space-y-1 text-xs text-slate-300">
                    <div>output dims: {thermalDebug.outputDims.join("x")}</div>
                    <div>hasObjectness: {String(thermalDebug.effectiveHasObjectness)}</div>
                    <div>boxFormat: {thermalDebug.effectiveBoxFormat}</div>
                    <div>
                      classes: start={thermalDebug.classStart}, num={thermalDebug.numClasses}
                    </div>
                    <div>returned boxes: {thermalDebug.returnedBoxes}</div>
                    <div>
                      fallback: {thermalDebug.fallbackUsed ? "used" : "not used"}
                      {thermalDebug.fallbackTried && thermalDebug.fallbackTried.length > 0 && (
                        <div className="mt-2 space-y-0.5">
                          {thermalDebug.fallbackTried.map((t, idx) => (
                            <div key={idx}>
                              try hasObj={String(t.hasObjectness)} box={t.boxFormat} boxes={t.returnedBoxes} thr=
                              {t.scoreThresholdUsed.toFixed(3)}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    <div>
                      Top5:
                      <div className="mt-1 space-y-0.5">
                        {thermalDebug.topBoxes.length === 0
                          ? "无"
                          : thermalDebug.topBoxes.map((b, idx) => (
                              <div key={idx}>
                                {b.className} {Math.round(b.score * 100)}%
                              </div>
                            ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        {errorText && <p className="text-sm text-rose-400">{errorText}</p>}
      </section>
    </main>
  );
}
