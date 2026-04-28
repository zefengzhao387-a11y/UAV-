"use client";

import * as ort from "onnxruntime-web";

export const MODEL_INPUT_SIZE = 640;

export interface DetectionBox {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
  score: number;
  classId: number;
  className: string;
}

export interface PreprocessMeta {
  scale: number;
  offsetX: number;
  offsetY: number;
  srcWidth: number;
  srcHeight: number;
  inputSize: number;
}

export interface PreprocessResult {
  tensor: ort.Tensor;
  meta: PreprocessMeta;
}

export interface AnalyzeOptions {
  modelPath?: string;
  classNames?: string[];
  hasObjectness?: boolean;
  applySigmoid?: boolean;
  fallbackToAlternateHead?: boolean;
  minBoxSize?: number;
  maxBoxAreaRatio?: number;
  scoreThreshold?: number;
  iouThreshold?: number;
  maxDetections?: number;
}

const DEFAULT_CLASS_NAMES = ["Defect"];
const DEFAULT_MODEL_PATH = "/model/yolov11s-pv.onnx";

const sessionPromiseMap = new Map<string, Promise<ort.InferenceSession>>();
const cachedSessionMap = new Map<string, ort.InferenceSession>();

/**
 * 加载 ONNX 模型（单例）。
 * 使用 Promise 缓存避免重复初始化 WebAssembly 与重复读模型。
 */
export async function loadModelSession(
  modelPath = DEFAULT_MODEL_PATH,
  onStatusChange?: (statusText: string) => void
): Promise<ort.InferenceSession> {
  const cachedSession = cachedSessionMap.get(modelPath);
  if (cachedSession) {
    return cachedSession;
  }

  if (!sessionPromiseMap.has(modelPath)) {
    const promise = (async () => {
      onStatusChange?.("正在初始化推理引擎...");
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
      ort.env.wasm.simd = true;

      onStatusChange?.("正在加载模型文件...");
      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
      cachedSessionMap.set(modelPath, session);
      onStatusChange?.("模型加载完成");
      return session;
    })().catch((error) => {
      sessionPromiseMap.delete(modelPath);
      throw error;
    });
    sessionPromiseMap.set(modelPath, promise);
  }

  return sessionPromiseMap.get(modelPath)!;
}

/**
 * 对图片进行缩放 + 居中裁剪到 640x640，再转成 NCHW Float32 Tensor。
 * 注意：
 * 1) 输出 shape 固定为 [1, 3, 640, 640]
 * 2) 像素归一化到 [0,1]，避免模型输入范围错误
 */
export function preprocessImage(imageElement: HTMLImageElement): PreprocessResult {
  const inputSize = MODEL_INPUT_SIZE;
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("无法创建离屏 Canvas 上下文");
  }

  const srcWidth = imageElement.naturalWidth || imageElement.width;
  const srcHeight = imageElement.naturalHeight || imageElement.height;
  const scale = Math.max(inputSize / srcWidth, inputSize / srcHeight);
  const resizedWidth = srcWidth * scale;
  const resizedHeight = srcHeight * scale;
  const offsetX = (inputSize - resizedWidth) / 2;
  const offsetY = (inputSize - resizedHeight) / 2;

  ctx.clearRect(0, 0, inputSize, inputSize);
  ctx.drawImage(imageElement, offsetX, offsetY, resizedWidth, resizedHeight);

  const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
  const { data } = imageData;

  const channelSize = inputSize * inputSize;
  const floatData = new Float32Array(3 * channelSize);

  // 按 NCHW 顺序写入 [R(0..n), G(0..n), B(0..n)]
  for (let i = 0; i < channelSize; i += 1) {
    const pixelIndex = i * 4;
    floatData[i] = data[pixelIndex] / 255.0;
    floatData[i + channelSize] = data[pixelIndex + 1] / 255.0;
    floatData[i + channelSize * 2] = data[pixelIndex + 2] / 255.0;
  }

  const tensor = new ort.Tensor("float32", floatData, [1, 3, inputSize, inputSize]);

  return {
    tensor,
    meta: {
      scale,
      offsetX,
      offsetY,
      srcWidth,
      srcHeight,
      inputSize
    }
  };
}

function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.min(Math.max(value, minValue), maxValue);
}

function intersectionOverUnion(a: DetectionBox, b: DetectionBox): number {
  const xA = Math.max(a.xMin, b.xMin);
  const yA = Math.max(a.yMin, b.yMin);
  const xB = Math.min(a.xMax, b.xMax);
  const yB = Math.min(a.yMax, b.yMax);

  const interWidth = Math.max(0, xB - xA);
  const interHeight = Math.max(0, yB - yA);
  const interArea = interWidth * interHeight;
  if (interArea <= 0) return 0;

  const areaA = Math.max(0, a.xMax - a.xMin) * Math.max(0, a.yMax - a.yMin);
  const areaB = Math.max(0, b.xMax - b.xMin) * Math.max(0, b.yMax - b.yMin);
  const union = areaA + areaB - interArea;
  if (union <= 0) return 0;
  return interArea / union;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

/**
 * 将模型坐标系(640x640裁剪图)映射回原图坐标。
 */
function remapToOriginal(box: DetectionBox, meta: PreprocessMeta): DetectionBox {
  const xMin = clamp((box.xMin - meta.offsetX) / meta.scale, 0, meta.srcWidth);
  const yMin = clamp((box.yMin - meta.offsetY) / meta.scale, 0, meta.srcHeight);
  const xMax = clamp((box.xMax - meta.offsetX) / meta.scale, 0, meta.srcWidth);
  const yMax = clamp((box.yMax - meta.offsetY) / meta.scale, 0, meta.srcHeight);

  return { ...box, xMin, yMin, xMax, yMax };
}

/**
 * 解析 YOLO 输出（兼容 [1,8400,84] 与 [1,84,8400] 两种布局）并执行 NMS。
 */
export function postprocessDetections(
  outputTensor: ort.Tensor,
  meta: PreprocessMeta,
  classNames: string[],
  hasObjectness: boolean,
  applySigmoid: boolean,
  minBoxSize: number,
  maxBoxAreaRatio: number,
  scoreThreshold = 0.5,
  iouThreshold = 0.45,
  maxDetections = 100
): DetectionBox[] {
  const raw = outputTensor.data as Float32Array;
  const dims = outputTensor.dims;
  if (dims.length !== 3) {
    throw new Error(`模型输出维度异常，期望 3 维，实际为 ${dims.length} 维`);
  }

  const [, d1, d2] = dims;
  // YOLO 常见输出为 [1, 8400, 84] 或 [1, 84, 8400]。
  // 这里统一按较小维度当作 featureLen（例如 84），较大维度当作候选框数量（例如 8400）。
  const featureLen = Math.min(d1, d2);
  const numBoxes = Math.max(d1, d2);
  const featuresOnD1 = d1 === featureLen;

  if (featureLen < 5) {
    throw new Error(`模型输出特征长度异常：${featureLen}`);
  }

  const read = (boxIndex: number, feature: number): number => {
    if (featuresOnD1) {
      return raw[feature * d2 + boxIndex];
    }
    return raw[boxIndex * d2 + feature];
  };

  // 兼容两种头：
  // 1) [cx, cy, w, h, cls...]
  // 2) [cx, cy, w, h, obj, cls...]
  const classStart = hasObjectness ? 5 : 4;
  const numClasses = featureLen - classStart;
  if (numClasses <= 0) {
    throw new Error(`模型类别维度异常：featureLen=${featureLen}`);
  }

  const candidates: DetectionBox[] = [];
  for (let i = 0; i < numBoxes; i += 1) {
    const cx = read(i, 0);
    const cy = read(i, 1);
    const w = read(i, 2);
    const h = read(i, 3);
    if (w <= 0 || h <= 0) continue;

    let bestClassId = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let c = 0; c < numClasses; c += 1) {
      const rawClsScore = read(i, classStart + c);
      const clsScore = applySigmoid ? sigmoid(rawClsScore) : rawClsScore;
      if (clsScore > bestScore) {
        bestScore = clsScore;
        bestClassId = c;
      }
    }

    if (hasObjectness) {
      const rawObjScore = read(i, 4);
      const objScore = applySigmoid ? sigmoid(rawObjScore) : rawObjScore;
      bestScore *= objScore;
    }

    if (!Number.isFinite(bestScore) || bestScore < scoreThreshold) continue;

    const rawBox: DetectionBox = {
      xMin: cx - w / 2,
      yMin: cy - h / 2,
      xMax: cx + w / 2,
      yMax: cy + h / 2,
      score: bestScore,
      classId: bestClassId,
      className: classNames[bestClassId] ?? `Class-${bestClassId}`
    };
    const mapped = remapToOriginal(rawBox, meta);
    const mappedW = mapped.xMax - mapped.xMin;
    const mappedH = mapped.yMax - mapped.yMin;
    if (mappedW < minBoxSize || mappedH < minBoxSize) continue;
    const mappedAreaRatio = (mappedW * mappedH) / (meta.srcWidth * meta.srcHeight);
    if (mappedAreaRatio > maxBoxAreaRatio) continue;
    if (mapped.xMax <= mapped.xMin || mapped.yMax <= mapped.yMin) continue;
    candidates.push(mapped);
  }

  candidates.sort((a, b) => b.score - a.score);
  const selected: DetectionBox[] = [];

  while (candidates.length > 0) {
    const current = candidates.shift();
    if (!current) break;
    selected.push(current);

    for (let i = candidates.length - 1; i >= 0; i -= 1) {
      const iou = intersectionOverUnion(current, candidates[i]);
      if (iou > iouThreshold) {
        candidates.splice(i, 1);
      }
    }
  }

  return selected.slice(0, maxDetections);
}

/**
 * 模型推理入口：预处理 -> session.run -> 后处理。
 */
export async function runYoloInference(
  imageElement: HTMLImageElement,
  options: AnalyzeOptions = {},
  onStatusChange?: (statusText: string) => void
): Promise<DetectionBox[]> {
  const modelPath = options.modelPath ?? DEFAULT_MODEL_PATH;
  const classNames = options.classNames ?? DEFAULT_CLASS_NAMES;
  const hasObjectness = options.hasObjectness ?? false;
  const applySigmoid = options.applySigmoid ?? false;
  const fallbackToAlternateHead = options.fallbackToAlternateHead ?? true;
  const minBoxSize = options.minBoxSize ?? 6;
  const maxBoxAreaRatio = options.maxBoxAreaRatio ?? 1.0;
  const scoreThreshold = options.scoreThreshold ?? 0.65;
  const iouThreshold = options.iouThreshold ?? 0.5;
  const maxDetections = options.maxDetections ?? 80;

  onStatusChange?.("正在加载模型...");
  const session = await loadModelSession(modelPath, onStatusChange);

  onStatusChange?.("正在执行图像预处理...");
  const { tensor, meta } = preprocessImage(imageElement);

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName || !outputName) {
    throw new Error("模型输入/输出名称解析失败，请检查 ONNX 文件");
  }

  onStatusChange?.("正在执行推理...");
  const results = await session.run({ [inputName]: tensor });
  const output = results[outputName];
  if (!output) {
    throw new Error(`未获取到模型输出：${outputName}`);
  }

  onStatusChange?.("正在后处理检测框...");
  const primaryBoxes = postprocessDetections(
    output,
    meta,
    classNames,
    hasObjectness,
    applySigmoid,
    minBoxSize,
    maxBoxAreaRatio,
    scoreThreshold,
    iouThreshold,
    maxDetections
  );

  // 红外热点模型导出格式可能不稳定（部分模型带 obj，部分不带）。
  // 若主策略没有任何框，自动尝试另一种头解析作为兜底，避免“完全无框”。
  if (fallbackToAlternateHead && primaryBoxes.length === 0) {
    const fallbackBoxes = postprocessDetections(
      output,
      meta,
      classNames,
      !hasObjectness,
      applySigmoid,
      minBoxSize,
      maxBoxAreaRatio,
      Math.max(0.25, scoreThreshold * 0.75),
      iouThreshold,
      maxDetections
    );
    if (fallbackBoxes.length > 0) {
      return fallbackBoxes;
    }
  }

  return primaryBoxes;
}

/**
 * 在叠加 Canvas 上绘制检测框与标签。
 */
export function drawBoundingBoxes(
  canvas: HTMLCanvasElement,
  imageElement: HTMLImageElement,
  boxes: DetectionBox[]
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const width = imageElement.clientWidth;
  const height = imageElement.clientHeight;
  const naturalWidth = imageElement.naturalWidth || width;
  const naturalHeight = imageElement.naturalHeight || height;

  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);

  // image 使用 object-contain 时，真实图像会在容器内产生留边（letterbox）。
  // 需按“实际绘制区域 + 偏移”映射，否则框会整体错位。
  const containScale = Math.min(width / naturalWidth, height / naturalHeight);
  const renderedWidth = naturalWidth * containScale;
  const renderedHeight = naturalHeight * containScale;
  const offsetX = (width - renderedWidth) / 2;
  const offsetY = (height - renderedHeight) / 2;

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ff4d4f";
  ctx.font = "12px sans-serif";
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const x = offsetX + box.xMin * containScale;
    const y = offsetY + box.yMin * containScale;
    const w = (box.xMax - box.xMin) * containScale;
    const h = (box.yMax - box.yMin) * containScale;
    const label = `${box.className}: ${(box.score * 100).toFixed(0)}%`;

    ctx.strokeRect(x, y, w, h);

    const textWidth = ctx.measureText(label).width;
    const labelHeight = 18;
    const labelY = Math.max(0, y - labelHeight - 2);
    ctx.fillStyle = "rgba(255, 77, 79, 0.75)";
    ctx.fillRect(x, labelY, textWidth + 10, labelHeight);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 5, labelY + 3);
  });
}
