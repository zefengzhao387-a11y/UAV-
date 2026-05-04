import type {
  AnalyzeOptions,
  DetectionBox,
  PostprocessDebugInfo
} from "@/utils/yoloTypes";

/**
 * 直连 Render 的 origin（可为空）。
 * @deprecated 推荐使用同源 /api/infer-proxy，见 NEXT_PUBLIC_USE_INFER_PROXY（默认开启）。
 */
export const INFER_SERVICE_URL = (process.env.NEXT_PUBLIC_INFER_SERVICE_URL ?? "").replace(
  /\/$/,
  ""
);

/** 默认启用同源代理，规避浏览器 Failed to fetch（跨域）。设为 0 / false / no 关闭。 */
export function useInferProxy(): boolean {
  const raw = process.env.NEXT_PUBLIC_USE_INFER_PROXY;
  if (raw === undefined || raw === "") return true;
  const v = raw.toLowerCase().trim();
  return v !== "0" && v !== "false" && v !== "no";
}

export function inferForwardingDescription(): string {
  const px = useInferProxy();
  if (!px) return INFER_SERVICE_URL || "(未配置)";
  const target = INFER_SERVICE_URL.trim() ? INFER_SERVICE_URL : "Render（请配置 NEXT_PUBLIC_INFER_SERVICE_URL）";
  return `/api/infer-proxy → ${target}`;
}

export function inferServiceConfigured(): boolean {
  if (useInferProxy()) return true;
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
  const proxied = useInferProxy();
  const base = INFER_SERVICE_URL.trim();

  if (proxied && !base) {
    throw new Error(
      "同源代理需在环境变量 NEXT_PUBLIC_INFER_SERVICE_URL（或服务端 INFER_SERVICE_URL）填写 Render 根地址以便转发"
    );
  }

  if (!proxied && !base) {
    throw new Error("未配置推理服务地址：NEXT_PUBLIC_USE_INFER_PROXY 关闭时请设置 NEXT_PUBLIC_INFER_SERVICE_URL");
  }

  onStatusChange?.("正在上传到推理服务...");
  const form = new FormData();
  form.append("image", imageFile, imageFile.name || "upload");
  form.append("model", kind);
  form.append("debug", debug ? "true" : "false");
  const payload = JSON.stringify(optionsToJson(options));
  form.append("options", payload);

  const url = proxied ? "/api/infer-proxy" : `${base}/detect`;

  const ac = new AbortController();
  const timer = globalThis.setTimeout(() => ac.abort(), 180000);
  try {
    const res = await fetch(url, {
      method: "POST",
      mode: proxied ? "same-origin" : "cors",
      credentials: "omit",
      body: form,
      signal: ac.signal
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
  } finally {
    globalThis.clearTimeout(timer);
  }
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
