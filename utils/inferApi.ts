import type {
  AnalyzeOptions,
  DetectionBox,
  PostprocessDebugInfo
} from "@/utils/yoloTypes";

/**
 * 公开推理服务的 origin（无尾部斜杠），例如 https://infer.example.com。
 * 构建时必填；需在 Vercel / 本地 `.env*` 配置 `NEXT_PUBLIC_INFER_SERVICE_URL`。
 */
export const INFER_SERVICE_URL = (process.env.NEXT_PUBLIC_INFER_SERVICE_URL ?? "").replace(
  /\/$/,
  ""
);

export function inferServiceConfigured(): boolean {
  return Boolean(INFER_SERVICE_URL.trim());
}

function optionsToJson(opts: AnalyzeOptions): Record<string, unknown> {
  const {
    classNames,
    hasObjectness,
    applySigmoid,
    fallbackToAlternateHead,
    minBoxSize,
    maxBoxAreaRatio,
    scoreThreshold,
    iouThreshold,
    maxDetections,
    boxFormat
  } = opts;

  const out: Record<string, unknown> = {};
  if (classNames !== undefined) out.classNames = classNames;
  if (hasObjectness !== undefined) out.hasObjectness = hasObjectness;
  if (applySigmoid !== undefined) out.applySigmoid = applySigmoid;
  if (fallbackToAlternateHead !== undefined) out.fallbackToAlternateHead = fallbackToAlternateHead;
  if (minBoxSize !== undefined) out.minBoxSize = minBoxSize;
  if (maxBoxAreaRatio !== undefined) out.maxBoxAreaRatio = maxBoxAreaRatio;
  if (scoreThreshold !== undefined) out.scoreThreshold = scoreThreshold;
  if (iouThreshold !== undefined) out.iouThreshold = iouThreshold;
  if (maxDetections !== undefined) out.maxDetections = maxDetections;
  if (boxFormat !== undefined) out.boxFormat = boxFormat;

  return out;
}

async function inferRemoteImpl(
  imageFile: File,
  kind: "visible" | "thermal",
  options: AnalyzeOptions,
  debug: boolean,
  onStatusChange?: (statusText: string) => void
): Promise<{ boxes: DetectionBox[]; debug?: PostprocessDebugInfo }> {
  const base = INFER_SERVICE_URL.trim();
  if (!base) {
    throw new Error("未配置推理服务地址：请在环境变量中设置 NEXT_PUBLIC_INFER_SERVICE_URL");
  }

  onStatusChange?.("正在上传到推理服务...");
  const form = new FormData();
  form.append("image", imageFile, imageFile.name || "upload");
  form.append("model", kind);
  form.append("debug", debug ? "true" : "false");
  const payload = JSON.stringify(optionsToJson(options));
  form.append("options", payload);

  const res = await fetch(`${base}/detect`, {
    method: "POST",
    body: form
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`推理服务返回 ${res.status}: ${text || res.statusText}`);
  }

  onStatusChange?.("正在解析检测结果...");
  const data = (await res.json()) as { boxes?: DetectionBox[]; debug?: PostprocessDebugInfo };
  const boxes = data.boxes;
  if (!Array.isArray(boxes)) {
    throw new Error("推理服务响应缺少 boxes");
  }

  return { boxes, ...(data.debug ? { debug: data.debug } : {}) };
}

export async function runYoloInferRemote(
  imageFile: File,
  kind: "visible" | "thermal",
  options: AnalyzeOptions,
  debug: boolean,
  onStatusChange?: (statusText: string) => void
): Promise<{ boxes: DetectionBox[]; debug?: PostprocessDebugInfo }> {
  return inferRemoteImpl(imageFile, kind, options, debug, onStatusChange);
}
