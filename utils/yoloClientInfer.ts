/**
 * 浏览器 ONNXRuntime Web 推理，预处理 / 后处理与 backend/yolo_infer.py 对齐。
 */

import type {
  AnalyzeOptions,
  DetectionBox,
  PostprocessDebugInfo
} from "@/utils/yoloTypes";
import type { InferenceSession, Tensor } from "onnxruntime-web";

const MODEL_INPUT_SIZE = 640;
const ONNXRUNTIME_WEB_VERSION = "1.20.1";
const WASM_CDN_ROOT = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNXRUNTIME_WEB_VERSION}/dist/`;

/** 可与 backend 文件名一致放置于 public/model/ */
export function visibleOnnxUrl(): string {
  const u = process.env.NEXT_PUBLIC_VISIBLE_MODEL_URL?.trim();
  return u || "/model/yolov11s-pv.onnx";
}

export function thermalOnnxUrl(): string {
  const u = process.env.NEXT_PUBLIC_THERMAL_MODEL_URL?.trim();
  return u || "/model/thermal-hotspot.onnx";
}

export interface PreprocessMeta {
  scale: number;
  offsetX: number;
  offsetY: number;
  srcWidth: number;
  srcHeight: number;
  inputSize: number;
}

interface BoxInternal {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
  score: number;
  classId: number;
  className: string;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function intersectionOverUnion(a: BoxInternal, b: BoxInternal): number {
  const x1 = Math.max(a.xMin, b.xMin);
  const y1 = Math.max(a.yMin, b.yMin);
  const x2 = Math.min(a.xMax, b.xMax);
  const y2 = Math.min(a.yMax, b.yMax);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const aa = Math.max(0, a.xMax - a.xMin) * Math.max(0, a.yMax - a.yMin);
  const bb = Math.max(0, b.xMax - b.xMin) * Math.max(0, b.yMax - b.yMin);
  const uni = aa + bb - inter;
  return uni > 0 ? inter / uni : 0;
}

function remapToOriginal(box: BoxInternal, meta: PreprocessMeta): BoxInternal {
  return {
    ...box,
    xMin: clamp((box.xMin - meta.offsetX) / meta.scale, 0, meta.srcWidth),
    yMin: clamp((box.yMin - meta.offsetY) / meta.scale, 0, meta.srcHeight),
    xMax: clamp((box.xMax - meta.offsetX) / meta.scale, 0, meta.srcWidth),
    yMax: clamp((box.yMax - meta.offsetY) / meta.scale, 0, meta.srcHeight)
  };
}

function readRaw(
  raw: Float32Array,
  flatCols: number,
  boxIndex: number,
  featureIndex: number,
  featuresOnD1: boolean
): number {
  if (featuresOnD1) {
    return raw[featureIndex * flatCols + boxIndex];
  }
  return raw[boxIndex * flatCols + featureIndex];
}

type BoxFmt = "cxcywh" | "xyxy";

function postprocessOnce(
  output3d: Float32Array,
  dims: readonly number[],
  meta: PreprocessMeta,
  classNames: string[],
  hasObjectness: boolean | "auto",
  applySigmoid: boolean,
  minBoxSize: number,
  maxBoxAreaRatio: number,
  scoreThreshold: number,
  iouThreshold: number,
  maxDetections: number,
  boxFormat: BoxFmt | "auto"
): BoxInternal[] {
  if (dims.length !== 3) throw new Error(`期望输出 3 维，当前 ${dims.length}`);

  const d1 = dims[1];
  const d2 = dims[2];
  const featureLen = Math.min(d1, d2);
  const numBoxes = Math.max(d1, d2);
  const flatCols = d2;
  const featuresOnD1 = d1 === featureLen;

  if (featureLen < 5) throw new Error(`特征维度异常：${featureLen}`);

  const pickHasObjectness = (): boolean => {
    if (hasObjectness !== "auto") return Boolean(hasObjectness);
    const expected = classNames.length;
    const noc = featureLen - 4;
    const woc = featureLen - 5;
    const diffNo = Math.abs(noc - expected);
    const diffWo = Math.abs(woc - expected);
    if (noc === expected && woc !== expected) return false;
    if (woc === expected && noc !== expected) return true;
    return diffWo <= diffNo;
  };

  const effectiveHasObj = pickHasObjectness();
  const classStart = effectiveHasObj ? 5 : 4;
  const numClasses = featureLen - classStart;
  if (numClasses <= 0) throw new Error(`类别维度异常 feature_len=${featureLen}`);

  const coordinateScale = 1;

  const runOnce = (fmt: BoxFmt): BoxInternal[] => {
    const candidates: BoxInternal[] = [];
    for (let i = 0; i < numBoxes; i += 1) {
      const cx =
        readRaw(output3d, flatCols, i, 0, featuresOnD1) *
        coordinateScale;
      const cy =
        readRaw(output3d, flatCols, i, 1, featuresOnD1) *
        coordinateScale;
      const ww =
        readRaw(output3d, flatCols, i, 2, featuresOnD1) *
        coordinateScale;
      const hh =
        readRaw(output3d, flatCols, i, 3, featuresOnD1) *
        coordinateScale;

      let xMin: number;
      let yMin: number;
      let xMax: number;
      let yMax: number;

      if (fmt === "cxcywh") {
        if (ww <= 0 || hh <= 0) continue;
        xMin = cx - ww / 2;
        yMin = cy - hh / 2;
        xMax = cx + ww / 2;
        yMax = cy + hh / 2;
      } else {
        if (ww <= cx || hh <= cy) continue;
        xMin = cx;
        yMin = cy;
        xMax = ww;
        yMax = hh;
      }

      let bestClassId = 0;
      let bestScore = -Infinity;
      for (let c = 0; c < numClasses; c += 1) {
        const rawCls = readRaw(
          output3d,
          flatCols,
          i,
          classStart + c,
          featuresOnD1
        );
        const clsScore = applySigmoid ? sigmoid(rawCls) : rawCls;
        if (clsScore > bestScore) {
          bestScore = clsScore;
          bestClassId = c;
        }
      }

      if (effectiveHasObj) {
        const rawObj = readRaw(output3d, flatCols, i, 4, featuresOnD1);
        const objScore = applySigmoid ? sigmoid(rawObj) : rawObj;
        bestScore *= objScore;
      }

      if (!Number.isFinite(bestScore) || bestScore < scoreThreshold) continue;
      if (xMax <= xMin || yMax <= yMin) continue;

      let rawBox: BoxInternal = {
        xMin,
        yMin,
        xMax,
        yMax,
        score: bestScore,
        classId: bestClassId,
        className:
          bestClassId < classNames.length
            ? classNames[bestClassId]
            : `Class-${bestClassId}`
      };
      const mapped = remapToOriginal(rawBox, meta);
      const mw = mapped.xMax - mapped.xMin;
      const mh = mapped.yMax - mapped.yMin;
      if (mw < minBoxSize || mh < minBoxSize) continue;
      const areaRatio = (mw * mh) / (meta.srcWidth * meta.srcHeight);
      if (areaRatio > maxBoxAreaRatio) continue;
      if (mapped.xMax <= mapped.xMin || mapped.yMax <= mapped.yMin) continue;
      candidates.push(mapped);
    }

    candidates.sort((a, b) => b.score - a.score);
    const selected: BoxInternal[] = [];
    while (candidates.length > 0) {
      const current = candidates.shift()!;
      selected.push(current);
      const remain = candidates.filter(
        (b) => intersectionOverUnion(current, b) <= iouThreshold
      );
      candidates.length = 0;
      candidates.push(...remain);
    }
    return selected.slice(0, maxDetections);
  };

  if (boxFormat === "auto") {
    const cxBoxes = runOnce("cxcywh");
    const xyBoxes = runOnce("xyxy");
    const ratio = (boxes: BoxInternal[]) =>
      boxes.reduce((s, b) => {
        const ww = Math.max(0, b.xMax - b.xMin);
        const hh = Math.max(0, b.yMax - b.yMin);
        return s + (ww * hh) / (meta.srcWidth * meta.srcHeight);
      }, 0) -
      boxes.length * 1e-5;

    const cxScore = ratio(cxBoxes);
    const xyScore = ratio(xyBoxes);
    return xyScore > cxScore ? xyBoxes : cxBoxes;
  }

  return runOnce(boxFormat);
}

function totalAreaRatio(boxes: BoxInternal[], meta: PreprocessMeta): number {
  return boxes.reduce((s, b) => {
    const ww = Math.max(0, b.xMax - b.xMin);
    const hh = Math.max(0, b.yMax - b.yMin);
    return s + (ww * hh) / (meta.srcWidth * meta.srcHeight);
  }, 0);
}

async function preprocessFromBitmap(
  bmp: ImageBitmap,
  inputSize: number = MODEL_INPUT_SIZE
): Promise<{ tensorData: Float32Array; meta: PreprocessMeta }> {
  const srcW = bmp.width;
  const srcH = bmp.height;
  const scale = Math.max(inputSize / srcW, inputSize / srcH);
  const resizedW = Math.round(srcW * scale);
  const resizedH = Math.round(srcH * scale);
  const offsetX = (inputSize - resizedW) / 2;
  const offsetY = (inputSize - resizedH) / 2;

  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 不可用");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.drawImage(bmp, Math.round(offsetX), Math.round(offsetY), resizedW, resizedH);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  const chw = new Float32Array(3 * inputSize * inputSize);
  let p = 0;
  const plane = inputSize * inputSize;
  for (let y = 0; y < inputSize; y += 1) {
    for (let x = 0; x < inputSize; x += 1) {
      const i = (y * inputSize + x) * 4;
      chw[p] = data[i] / 255;
      chw[plane + p] = data[i + 1] / 255;
      chw[plane * 2 + p] = data[i + 2] / 255;
      p += 1;
    }
  }

  bmp.close?.();

  return {
    tensorData: chw,
    meta: {
      scale,
      offsetX,
      offsetY,
      srcWidth: srcW,
      srcHeight: srcH,
      inputSize
    }
  };
}

async function preprocessFromFile(
  imageFile: File,
  inputSize: number = MODEL_INPUT_SIZE
): Promise<{ tensorData: Float32Array; meta: PreprocessMeta }> {
  const bmp = await createImageBitmap(imageFile);
  return preprocessFromBitmap(bmp, inputSize);
}

type OrtNs = typeof import("onnxruntime-web");

let ortModule: OrtNs | undefined;

async function getOrt(): Promise<OrtNs> {
  if (typeof window === "undefined") {
    throw new Error("前端 ONNX 仅能在浏览器中使用");
  }
  if (!ortModule) {
    ortModule = await import("onnxruntime-web");
    const custom = process.env.NEXT_PUBLIC_ONNXRUNTIME_WASM_PATH?.trim();
    const wasmRoot = custom?.replace(/\/?$/, "/") || WASM_CDN_ROOT;
    ortModule.env.wasm.wasmPaths = wasmRoot.endsWith("/") ? wasmRoot : `${wasmRoot}/`;
    ortModule.env.wasm.numThreads = 1;
  }
  return ortModule;
}

const sessionByUrl = new Map<string, InferenceSession>();

async function getSessionForUrl(ort: OrtNs, url: string): Promise<InferenceSession> {
  if (sessionByUrl.has(url)) {
    return sessionByUrl.get(url)!;
  }
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(
      `加载 ONNX 失败 ${resp.status}：${url}。请将模型放到 public/model/ 或配置 NEXT_PUBLIC_*_MODEL_URL。`
    );
  }
  const buf = await resp.arrayBuffer();
  const session = await ort.InferenceSession.create(buf, {
    executionProviders: ["wasm"]
  });
  sessionByUrl.set(url, session);
  return session;
}

function toBoxesPublic(internal: BoxInternal[]): DetectionBox[] {
  return internal.map((b) => ({
    xMin: b.xMin,
    yMin: b.yMin,
    xMax: b.xMax,
    yMax: b.yMax,
    score: b.score,
    classId: b.classId,
    className: b.className
  }));
}

/** 与 yolo_infer.py：未传则按 False；仅 "auto" 时按维度猜测。 */
function resolveEffectiveHo(
  hasOpt: AnalyzeOptions["hasObjectness"],
  featureLen: number,
  expectedClasses: number
): boolean {
  if (hasOpt === "auto") {
    const noc = featureLen - 4;
    const woc = featureLen - 5;
    const expected = expectedClasses;
    const diffNo = Math.abs(noc - expected);
    const diffWo = Math.abs(woc - expected);
    if (noc === expected && woc !== expected) return false;
    if (woc === expected && noc !== expected) return true;
    return diffWo <= diffNo;
  }
  return Boolean(hasOpt);
}

function chooseFormatBoxes(
  out: Float32Array,
  outDims: number[],
  meta: PreprocessMeta,
  explicitHasObj: boolean,
  opts: {
    applySigmoid: boolean;
    minBoxSize: number;
    maxBoxAreaRatio: number;
    scoreThreshold: number;
    iouThreshold: number;
    maxDetections: number;
    boxFormatOpt: AnalyzeOptions["boxFormat"];
    classNames: string[];
  },
  threshold: number
): { fmt: BoxFmt | "mixed"; boxes: BoxInternal[] } {
  const {
    applySigmoid,
    minBoxSize,
    maxBoxAreaRatio,
    iouThreshold,
    maxDetections,
    boxFormatOpt,
    classNames
  } = opts;

  if (boxFormatOpt && boxFormatOpt !== "auto") {
    const bf: BoxFmt = boxFormatOpt === "xyxy" ? "xyxy" : "cxcywh";
    const boxes = postprocessOnce(
      out,
      outDims,
      meta,
      classNames,
      explicitHasObj,
      applySigmoid,
      minBoxSize,
      maxBoxAreaRatio,
      threshold,
      iouThreshold,
      maxDetections,
      bf
    );
    return { fmt: bf, boxes };
  }

  const cxB = postprocessOnce(
    out,
    outDims,
    meta,
    classNames,
    explicitHasObj,
    applySigmoid,
    minBoxSize,
    maxBoxAreaRatio,
    threshold,
    iouThreshold,
    maxDetections,
    "cxcywh"
  );
  const xyB = postprocessOnce(
    out,
    outDims,
    meta,
    classNames,
    explicitHasObj,
    applySigmoid,
    minBoxSize,
    maxBoxAreaRatio,
    threshold,
    iouThreshold,
    maxDetections,
    "xyxy"
  );
  const cxSc = totalAreaRatio(cxB, meta) - cxB.length * 1e-5;
  const xySc = totalAreaRatio(xyB, meta) - xyB.length * 1e-5;
  if (xySc > cxSc) return { fmt: "xyxy", boxes: xyB };
  return { fmt: "cxcywh", boxes: cxB };
}

function debugFromResult(
  outDims: readonly number[],
  boxes: DetectionBox[],
  effectiveHasObj: boolean,
  effectiveFmt: string,
  classStart: number,
  fbUsed: boolean,
  fbTried: NonNullable<PostprocessDebugInfo["fallbackTried"]> | undefined
): PostprocessDebugInfo {
  const d1 = outDims[1];
  const d2 = outDims[2];
  const featureLen = Math.min(d1!, d2!);
  const numBoxes = Math.max(d1!, d2!);
  const numClasses = featureLen - classStart;
  const top = [...boxes].sort((a, b) => b.score - a.score).slice(0, 5);
  return {
    outputDims: [...outDims],
    featureLen,
    numBoxes,
    coordinateScale: 1,
    effectiveHasObjectness: effectiveHasObj,
    effectiveBoxFormat: effectiveFmt as "cxcywh" | "xyxy",
    classStart,
    numClasses,
    returnedBoxes: boxes.length,
    topBoxes: top.map((b) => ({
      className: b.className,
      score: b.score,
      classId: b.classId
    })),
    fallbackUsed: fbUsed,
    ...(fbTried ? { fallbackTried: fbTried } : {})
  };
}

async function inferCore(
  imageFile: File,
  modelUrl: string,
  options: AnalyzeOptions,
  debug: boolean
): Promise<{ boxes: DetectionBox[]; debug?: PostprocessDebugInfo }> {
  const ort = await getOrt();
  const { tensorData, meta } = await preprocessFromFile(imageFile);
  const session = await getSessionForUrl(ort, modelUrl);
  const inName = session.inputNames[0];
  const outName = session.outputNames[0];
  const tensor = new ort.Tensor("float32", tensorData, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const feeds: Record<string, Tensor> = { [inName]: tensor };
  const results = await session.run(feeds);
  const outTensor = results[outName];
  const outData = outTensor.data as Float32Array;
  const dims = [...outTensor.dims];
  if (dims.length !== 3) throw new Error("模型输出维度异常");

  const classNames = options.classNames ?? [];
  const applySigmoid = Boolean(options.applySigmoid);
  const minBoxSize = options.minBoxSize ?? 6;
  const maxBoxAreaRatio = options.maxBoxAreaRatio ?? 1;
  const scoreThreshold = options.scoreThreshold ?? 0.65;
  const iouThreshold = options.iouThreshold ?? 0.5;
  const maxDetections = options.maxDetections ?? 80;
  const boxFormatOpt = options.boxFormat ?? "cxcywh";
  const fallbackToAlternateHead = options.fallbackToAlternateHead !== false;

  const featureLen = Math.min(dims[1]!, dims[2]!);
  const effPrimaryHo = resolveEffectiveHo(
    options.hasObjectness,
    featureLen,
    classNames.length
  );

  const primaryChoose = chooseFormatBoxes(
    outData,
    dims,
    meta,
    effPrimaryHo,
    {
      applySigmoid,
      minBoxSize,
      maxBoxAreaRatio,
      scoreThreshold,
      iouThreshold,
      maxDetections,
      boxFormatOpt,
      classNames
    },
    scoreThreshold
  );

  let fbUsed = false;
  let fbTried: NonNullable<PostprocessDebugInfo["fallbackTried"]> | undefined;

  let finalBoxes = primaryChoose.boxes;
  let finalHo = effPrimaryHo;
  let finalFmt =
    typeof primaryChoose.fmt === "string" && primaryChoose.fmt !== "mixed"
      ? primaryChoose.fmt
      : "cxcywh";

  if (fallbackToAlternateHead && primaryChoose.boxes.length === 0) {
    fbUsed = true;
    fbTried = [];
    const fbThr = Math.max(0.25, scoreThreshold * 0.75);
    const hoOpt = options.hasObjectness ?? false;
    const orders =
      hoOpt === "auto" ? [false, true] : [!Boolean(hoOpt)];

    for (const obj of orders) {
      const r = chooseFormatBoxes(
        outData,
        dims,
        meta,
        obj,
        {
          applySigmoid,
          minBoxSize,
          maxBoxAreaRatio,
          scoreThreshold: fbThr,
          iouThreshold,
          maxDetections,
          boxFormatOpt,
          classNames
        },
        fbThr
      );
      fbTried.push({
        hasObjectness: obj,
        boxFormat: r.fmt === "xyxy" ? "xyxy" : "cxcywh",
        returnedBoxes: r.boxes.length,
        scoreThresholdUsed: fbThr
      });
      if (r.boxes.length > 0) {
        finalHo = obj;
        finalFmt = typeof r.fmt === "string" && r.fmt !== "mixed" ? r.fmt : finalFmt;
        finalBoxes = r.boxes;
        break;
      }
    }
  }

  const publicBoxes = toBoxesPublic(finalBoxes);
  const classStart = finalHo ? 5 : 4;

  if (!debug) {
    return { boxes: publicBoxes };
  }

  return {
    boxes: publicBoxes,
    debug: debugFromResult(dims, publicBoxes, finalHo, finalFmt, classStart, fbUsed, fbTried)
  };
}

/** 远端推理开关关闭时使用；在后端 ONNX 与同版本 ORT-Web 对齐。 */
export async function runYoloInferClient(
  imageFile: File,
  kind: "visible" | "thermal",
  options: AnalyzeOptions,
  debug: boolean,
  onStatusChange?: (t: string) => void
): Promise<{ boxes: DetectionBox[]; debug?: PostprocessDebugInfo }> {
  const modelUrl = options.modelPath?.trim()
    ? options.modelPath!.replace(/\\/g, "/")
    : kind === "visible"
      ? visibleOnnxUrl()
      : thermalOnnxUrl();

  onStatusChange?.("正在加载 ONNX 运行时…");
  onStatusChange?.(`正在推理（${kind}）…`);

  try {
    return await inferCore(imageFile, modelUrl, options, debug);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(
      `${msg}。建议使用与 backend 同款导出 ONNX；若为跨域 CSP，可把 WASM 拷贝到 public/ 并设 NEXT_PUBLIC_ONNXRUNTIME_WASM_PATH=/onnx-wasm/。`
    );
  }
}
