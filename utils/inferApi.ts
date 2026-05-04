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

/** 直连失败后是否自动改走 /api/infer-proxy（默认开启，可设 0 / false 关掉）。 */
function useProxyFailover(): boolean {
  const raw = process.env.NEXT_PUBLIC_INFER_PROXY_FAILOVER;
  if (raw === undefined || raw === "") return true;
  const v = raw.toLowerCase().trim();
  return v !== "0" && v !== "false" && v !== "no";
}

export function inferForwardingDescription(): string {
  const px = useInferProxy();
  const fo = !px && useProxyFailover();
  if (px) {
    const target = INFER_SERVICE_URL.trim() ? INFER_SERVICE_URL : "Render（请配置 NEXT_PUBLIC_INFER_SERVICE_URL）";
    return `/api/infer-proxy → ${target}`;
  }
  if (fo) {
    return `先直连 ${INFER_SERVICE_URL || "Render"}，失败则本站 /api/infer-proxy 兜底`;
  }
  return INFER_SERVICE_URL || "(未配置)";
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

/** 一次 POST：带重试；directRender 时对 502/CORS 级网络错误可多试几次。 */
async function postDetectWithRetries(
  targetUrl: string,
  sameOrigin: boolean,
  directRender: boolean,
  maxAttempts: number,
  buildFormData: () => FormData,
  onStatusChange?: (statusText: string) => void
): Promise<{ boxes: DetectionBox[]; debug?: PostprocessDebugInfo }> {
  let lastThrow: unknown;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const ac = new AbortController();
    const timer = globalThis.setTimeout(() => ac.abort(), 180000);
    try {
      if (attempt > 0 && directRender && maxAttempts > 1) {
        onStatusChange?.(
          `推理服务繁忙或冷启动中，正在进行第 ${attempt + 1} 次请求（最多 ${maxAttempts} 次）…`
        );
        await sleep(Math.min(45_000, 5000 + attempt * 8000));
      } else if (attempt > 0 && !directRender && maxAttempts > 1) {
        onStatusChange?.(`网关重试 ${attempt + 1}/${maxAttempts}…`);
        await sleep(Math.min(5000, 400 + attempt * 650));
      }

      const useBypassHeader = Boolean(sameOrigin && VERCEL_PROTECTION_BYPASS.trim());
      const headers: HeadersInit = useBypassHeader
        ? { "x-vercel-protection-bypass": VERCEL_PROTECTION_BYPASS.trim() }
        : {};

      const res = await fetch(targetUrl, {
        method: "POST",
        mode: sameOrigin ? "same-origin" : "cors",
        credentials: "omit",
        headers,
        body: buildFormData(),
        signal: ac.signal
      });

      if (!res.ok) {
        let text = await res.text();
        if (text.startsWith("{") && text.includes('"error"')) {
          try {
            const j = JSON.parse(text) as { error?: string; hint?: string };
            const parts = [j.error, j.hint].filter(Boolean);
            if (parts.length > 0) text = parts.join(" — ");
          } catch {
            /* keep raw text */
          }
        }
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

        const retry502 =
          directRender && DIRECT_RETRY_CODES.has(res.status) && attempt < maxAttempts - 1;
        const retryProxied502 =
          !directRender && DIRECT_RETRY_CODES.has(res.status) && attempt < maxAttempts - 1;
        if (retry502 || retryProxied502) {
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
      /** 直连易受 CORS/网关影响；同源代理仍可遇边缘节点瞬时失败，服务端已重试时再补一轮客户端 retry。 */
      const transientNetwork =
        e instanceof TypeError ||
        e instanceof DOMException ||
        (e instanceof Error &&
          (/Failed to fetch|NetworkError|Load failed/i.test(e.message) || e.name === "AbortError"));
      const transient = attempt < maxAttempts - 1 && transientNetwork;
      if (transient) {
        continue;
      }
      throw e;
    } finally {
      globalThis.clearTimeout(timer);
    }
  }

  if (lastThrow instanceof Error) throw lastThrow;
  throw new Error("推理请求多次重试仍未成功");
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
  const failover = useProxyFailover();

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
    form.append("options", JSON.stringify(optionsToJson(options)));
    return form;
  };

  const drParsed = Number(process.env.NEXT_PUBLIC_INFER_FETCH_RETRIES ?? "5");
  const directAttempts = Math.max(
    1,
    Number.isFinite(drParsed) ? Math.floor(drParsed) : 5
  );
  const proxyAttempts = proxied ? 3 : failover ? 4 : 2;

  onStatusChange?.("正在上传到推理服务…");

  if (proxied) {
    return postDetectWithRetries(
      "/api/infer-proxy",
      true,
      false,
      proxyAttempts,
      buildFormData,
      onStatusChange
    );
  }

  try {
    return await postDetectWithRetries(
      `${base}/detect`,
      false,
      true,
      directAttempts,
      buildFormData,
      onStatusChange
    );
  } catch (firstErr) {
    if (!failover || !base) throw firstErr;
    onStatusChange?.(
      "直连推理不稳定（常见：Render 休眠/502 且无跨域头），改经本站 /api/infer-proxy 转发…"
    );
    try {
      return await postDetectWithRetries(
        "/api/infer-proxy",
        true,
        false,
        proxyAttempts,
        buildFormData,
        onStatusChange
      );
    } catch (proxyErr) {
      throw proxyErr instanceof Error ? proxyErr : firstErr;
    }
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
