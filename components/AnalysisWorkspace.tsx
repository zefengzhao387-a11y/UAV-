"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ImageUploadCard from "@/components/ImageUploadCard";
import TopNav from "@/components/TopNav";
import { DetectionBox, drawBoundingBoxes, runYoloInference } from "@/utils/modelHelper";

const VISIBLE_MODEL_PATH = "/model/yolov11s-pv.onnx";
const THERMAL_MODEL_PATH = "/model/thermal-hotspot.onnx";
const VISIBLE_CLASSES = ["Clean", "Dust", "Bird", "Electrical", "Physical", "Snow"];
const THERMAL_CLASSES = ["Hotspot"];
const STRUCTURAL_DAMAGE_CLASSES = new Set([
  "physical",
  "electrical",
  "crack",
  "brokencell",
  "fragment",
  "star_crack"
]);
const HOTSPOT_CLASSES = new Set(["hotspot", "hot-spot", "hot_spot"]);
const CROSS_MODAL_IOU_THRESHOLD = 0.06;

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
  const [visibleFile, setVisibleFile] = useState<File | null>(null);
  const [thermalFile, setThermalFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusText, setStatusText] = useState("等待上传两张图像");
  const [errorText, setErrorText] = useState<string | null>(null);
  const [fusionResult, setFusionResult] = useState<FusionResult>({
    highRiskCount: 0,
    structuralCandidateCount: 0,
    hotspotCandidateCount: 0
  });
  const [visibleScoreThreshold, setVisibleScoreThreshold] = useState(0.52);
  const [visibleIouThreshold, setVisibleIouThreshold] = useState(0.55);
  const [thermalScoreThreshold, setThermalScoreThreshold] = useState(0.25);
  const [thermalIouThreshold, setThermalIouThreshold] = useState(0.4);

  const visibleUrl = useObjectUrl(visibleFile);
  const thermalUrl = useObjectUrl(thermalFile);

  const visibleImgRef = useRef<HTMLImageElement | null>(null);
  const thermalImgRef = useRef<HTMLImageElement | null>(null);
  const visibleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const thermalCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const canAnalyze = useMemo(
    () => Boolean(visibleUrl && thermalUrl) && !isAnalyzing,
    [visibleUrl, thermalUrl, isAnalyzing]
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

  const runAnalyze = async () => {
    if (!canAnalyze) return;
    setErrorText(null);
    setFusionResult({ highRiskCount: 0, structuralCandidateCount: 0, hotspotCandidateCount: 0 });
    setIsAnalyzing(true);

    try {
      const [visibleImage, thermalImage] = await Promise.all([
        waitImageReady(visibleImgRef),
        waitImageReady(thermalImgRef)
      ]);

      setStatusText("正在分析可见光图像...");
      const visibleBoxes = await runYoloInference(
        visibleImage,
        {
          modelPath: VISIBLE_MODEL_PATH,
          classNames: VISIBLE_CLASSES,
          hasObjectness: false,
          minBoxSize: 2,
          maxBoxAreaRatio: 0.8,
          scoreThreshold: visibleScoreThreshold,
          iouThreshold: visibleIouThreshold,
          maxDetections: 120
        },
        setStatusText
      );
      drawBoundingBoxes(visibleCanvasRef.current!, visibleImage, visibleBoxes);

      setStatusText("正在分析红外热力图...");
      const thermalBoxes = await runYoloInference(
        thermalImage,
        {
          modelPath: THERMAL_MODEL_PATH,
          classNames: THERMAL_CLASSES,
          hasObjectness: true,
          applySigmoid: true,
          fallbackToAlternateHead: false,
          minBoxSize: 8,
          maxBoxAreaRatio: 0.15,
          scoreThreshold: thermalScoreThreshold,
          iouThreshold: thermalIouThreshold,
          maxDetections: 20
        },
        setStatusText
      );
      drawBoundingBoxes(thermalCanvasRef.current!, thermalImage, thermalBoxes);

      // 双光因果闭环：
      // 仅当“结构性物理损伤”与“热斑”在归一化二维坐标上重叠触发，才确诊高危病灶。
      const structuralVisibleBoxes = visibleBoxes.filter((box) =>
        STRUCTURAL_DAMAGE_CLASSES.has(box.className.toLowerCase())
      );
      const hotspotBoxes = thermalBoxes.filter((box) =>
        HOTSPOT_CLASSES.has(box.className.toLowerCase())
      );

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
      setStatusText("分析失败，请检查模型与图像");
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    clearCanvas(visibleCanvasRef.current);
  }, [visibleUrl]);

  useEffect(() => {
    clearCanvas(thermalCanvasRef.current);
  }, [thermalUrl]);

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
          imageUrl={visibleUrl}
          onFileSelected={setVisibleFile}
          imageRef={visibleImgRef}
          canvasRef={visibleCanvasRef}
        />

        <ImageUploadCard
          title="红外热力图 (Thermal Image)"
          hint="用于识别热点与热异常区域"
          imageUrl={thermalUrl}
          onFileSelected={setThermalFile}
          imageRef={thermalImgRef}
          canvasRef={thermalCanvasRef}
        />
      </section>

      <section className="mt-7 flex flex-col items-center gap-3">
        <div className="panel w-full max-w-3xl px-4 py-4">
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

        <button
          type="button"
          disabled={!canAnalyze}
          onClick={runAnalyze}
          className="rounded-lg bg-cyan-500 px-8 py-3 font-medium text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
        >
          {isAnalyzing ? "分析中..." : "开始缺陷分析"}
        </button>
        <p className="text-sm text-slate-300">{statusText}</p>
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
        {errorText && <p className="text-sm text-rose-400">{errorText}</p>}
      </section>
    </main>
  );
}
