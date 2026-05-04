import type {
  AnalyzeOptions,
  DetectionBox,
  PostprocessDebugInfo
} from "@/utils/yoloTypes";

/**
 * 直连 Render 的 origin（可为空）。
 */
export const INFER_SERVICE_URL = (process.env.NEXT_PUBLIC_INFER_SERVICE_URL ?? "").replace(
  /\/$/,
  ""
);

/** Vercel 部署保护绕过（见文末注释）；留空则无。 */
const VERCEL_PROTECTION_BYPASS = (
  process.env.NEXT_PUBLIC_VERCEL_PROTECTION_BYPASS ?? ""
).trim();

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

const DIRECT_RETRY_CODES = new Set([502, 503, 504]);

function sleep(ms: number): Promise<void> {
  return new Promise((r) => {
    globalThis.setTimeout(r, ms);
  });
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

  const buildFormData = (): FormData => {
    const form = new FormData();
    form.append("image", imageFile, imageFile.name || "upload");
    form.append("model", kind);
    form.append("debug", debug ? "true" : "false");
    const payload = JSON.stringify(optionsToJson(options));
    form.append("options", payload);
    return form;
  };

  const url = proxied ? "/api/infer-proxy" : `${base}/detect`;
  const drParsed = Number(process.env.NEXT_PUBLIC_INFER_FETCH_RETRIES ?? "5");
  const maxAttempts = proxied
    ? 3
    : Math.max(1, Number.isFinite(drParsed) ? Math.floor(drParsed) : 5);

  onStatusChange?.("正在上传到推理服务...");

  let lastThrow: unknown;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const ac = new AbortController();
    const timer = globalThis.setTimeout(() => ac.abort(), 180000);
    try {
      if (attempt > 0 && !proxied && maxAttempts > 1) {
        onStatusChange?.(
          `推理服务繁忙或冷启动中，正在进行第 ${attempt + 1} 次请求（最多 ${maxAttempts} 次）…`
        );
        await sleep(Math.min(45_000, 5000 + attempt * 8000));
      }

      const headers: HeadersInit =
        proxied && VERCEL_PROTECTION_BYPASS
          ? { "x-vercel-protection-bypass": VERCEL_PROTECTION_BYPASS }
          : {};

      const res = await fetch(url, {
        method: "POST",
        mode: proxied ? "same-origin" : "cors",
        credentials: "omit",
        headers,
        body: buildFormData(),
        signal: ac.signal
      });

      if (!res.ok) {
        const text = await res.text();
        if (
          res.status === 401 &&
          (text.includes("Vercel Authentication") || text.includes("Authentication Required"))
        ) {
          throw new Error(
            "401：这是 Vercel「部署保护」（Deployment Protection）拦截了同源 /api 请求，不是 Render 报错。可选：① 使用 Production/"
              + "自定义正式域名访问；② 在 Vercel 关闭 Preview 的 Protection；③ 或使用 Automation Bypass Secret 填环境变量 "
              + "NEXT_PUBLIC_VERCEL_PROTECTION_BYPASS（会暴露在浏览器）。"
          );
        }
        if (!proxied && DIRECT_RETRY_CODES.has(res.status) && attempt < maxAttempts - 1) {
          lastThrow = new Error(`推理服务返回 ${res.status}: ${text.slice(0, 200)}`);
          continue;
        }
        throw new Error(`推理服务返回 ${res.status}: ${text || res.statusText}`);
      }

      onStatusChange?.("正在解析检测结果...");
      const data = (await res.json()) as {
        boxes?: DetectionBox[];
        debug?: PostprocessDebugInfo;
      };
      const boxes = data.boxes;
      if (!Array.isArray(boxes)) {
        throw new Error("推理服务响应缺少 boxes");
      }

      return { boxes, ...(data.debug ? { debug: data.debug } : {}) };
    } catch (e) {
      lastThrow = e;
      const transient =
        !proxied &&
        attempt < maxAttempts - 1 &&
        (e instanceof TypeError ||
          e instanceof DOMException ||
          (e instanceof Error &&
            (/Failed to fetch|NetworkError|Load failed/i.test(e.message) ||
              e.name === "AbortError")));

      if (transient) {
        continue;
      }
      throw e;
    } finally {
      globalThis.clearTimeout(timer);
    }
  }

  if (lastThrow instanceof Error) {
    throw lastThrow;
  }
  throw new Error("推理请求多次重试仍未成功");
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
