import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
/** Hobby 常为 60s；推理慢时可升级计划或拆分优化 */
export const maxDuration = 120;

/** 快照后可多次组装新的 FormData，避免 multipart body 仅能消费一次。 */
function snapshotFormData(source: FormData): [string, FormDataEntryValue][] {
  return Array.from(source.entries());
}

function formDataFromSnapshot(entries: readonly [string, FormDataEntryValue][]): FormData {
  const next = new FormData();
  for (const [k, v] of entries) next.append(k, v);
  return next;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => globalThis.setTimeout(r, ms));
}

function upstreamBase(): string {
  const raw =
    process.env.INFER_SERVICE_URL?.trim() ||
    process.env.NEXT_PUBLIC_INFER_SERVICE_URL?.trim() ||
    "";
  return raw.replace(/\/$/, "");
}

function envNum(name: string, fallback: number): number {
  const v = Number(process.env[name]);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

function truthyEnv(name: string, defaultTrue: boolean): boolean {
  const raw = process.env[name]?.toLowerCase().trim();
  if (raw === undefined || raw === "") return defaultTrue;
  return !(raw === "0" || raw === "false" || raw === "no");
}

const RETRY_STATUS = new Set([502, 503, 504]);

/** 服务端→Render：唤醒 + detect 预算内多次重试（比浏览器单层重试更适合冷实例 / 网关错误）。 */
export async function POST(request: NextRequest) {
  const base = upstreamBase();
  if (!base) {
    return NextResponse.json(
      {
        error:
          "服务端未配置 INFER_SERVICE_URL 或 NEXT_PUBLIC_INFER_SERVICE_URL，无法转发到 Render"
      },
      { status: 503 }
    );
  }

  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json({ error: "无效的 multipart 请求" }, { status: 400 });
  }

  const snapshot = snapshotFormData(formData);
  const wake = truthyEnv("INFER_PROXY_WAKE_HEALTH", true);
  const maxAttempts = Math.min(14, Math.max(1, Math.floor(envNum("INFER_PROXY_MAX_ATTEMPTS", 8))));
  const routeBudgetMs = Math.floor(envNum("INFER_PROXY_ROUTE_BUDGET_MS", 115000));
  const baseBackoffMs = Math.floor(envNum("INFER_PROXY_BACKOFF_MS", 3500));

  const deadline = Date.now() + routeBudgetMs;

  async function pokeHealth(reason: string): Promise<void> {
    if (!wake) return;
    const left = deadline - Date.now() - 2000;
    if (left < 2500) return;
    try {
      const ctl = AbortSignal.timeout(Math.min(28_000, left));
      const r = await fetch(`${base}/health`, { method: "GET", cache: "no-store", signal: ctl });
      if (!r.ok) {
        console.warn(`[infer-proxy] wake /health ${r.status} (${reason})`);
      }
    } catch (e) {
      console.warn(`[infer-proxy] wake /health failed (${reason}):`, e);
    }
  }

  await pokeHealth("pre-detect");

  let attempt = 0;
  let lastStatus = 502;
  let lastBody = new Uint8Array();
  let lastCt = "application/json";
  let lastErr: string | undefined;

  while (attempt < maxAttempts && Date.now() < deadline) {
    attempt += 1;
    const remaining = deadline - Date.now() - 1500;
    if (remaining < 6000) {
      lastErr = "infer-proxy：预算时间内无法完成推理（可考虑升级 Vercel 函数时长或减少重试间隔）";
      break;
    }

    const upstreamTimeoutMs = Math.min(110000, remaining);
    try {
      const ac = AbortSignal.timeout(upstreamTimeoutMs);
      const res = await fetch(`${base}/detect`, {
        method: "POST",
        body: formDataFromSnapshot(snapshot),
        signal: ac
      });

      const buf = new Uint8Array(await res.arrayBuffer());
      lastStatus = res.status;
      lastBody = buf;
      lastCt = res.headers.get("content-type")?.split(";")[0]?.trim() || "application/json";

      if (res.ok) {
        return new NextResponse(buf, {
          status: res.status,
          headers: { "Content-Type": lastCt }
        });
      }

      if (!RETRY_STATUS.has(res.status) || attempt >= maxAttempts) {
        return new NextResponse(buf, {
          status: res.status,
          headers: { "Content-Type": lastCt }
        });
      }

      const backoff = Math.min(22_000, baseBackoffMs * 2 ** Math.min(attempt - 1, 4));
      console.warn(
        `[infer-proxy] upstream ${res.status}, retry ${attempt}/${maxAttempts} after ${backoff}ms`
      );
      await sleep(backoff);
      await pokeHealth(`after-${res.status}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      lastErr = msg;
      const transientNetwork =
        e instanceof TypeError ||
        (e instanceof Error &&
          (e.name === "AbortError" ||
            /fetch failed|aborted|AbortError|timeout|ECONNRESET|ETIMEDOUT|ENOTFOUND|socket|EAI_AGAIN|network/i.test(
              msg
            )));
      const shouldGiveUp = attempt >= maxAttempts || !transientNetwork;
      if (shouldGiveUp) {
        return NextResponse.json(
          {
            error: `转发推理服务失败: ${msg}`,
            upstreamBase: base,
            attempts: attempt,
            maxAttempts
          },
          { status: 502 }
        );
      }
      const backoff = Math.min(22_000, baseBackoffMs * 2 ** Math.min(attempt - 1, 4));
      console.warn(`[infer-proxy] fetch threw (${msg}), retry ${attempt}/${maxAttempts} after ${backoff}ms`);
      await sleep(backoff);
      await pokeHealth("after-error");
    }
  }

  return NextResponse.json(
    {
      error:
        lastErr ??
        (lastBody.length
          ? `上游持续返回 ${lastStatus}：${new TextDecoder().decode(lastBody.slice(0, 280))}`
          : `上游在 ${attempt} 次尝试后仍不可用`),
      upstreamBase: base,
      attempts: attempt,
      maxAttempts,
      lastUpstreamStatus: lastStatus,
      hint:
        "多为 Render 冷启动、休眠或内存不足。可在 Render 打开实例并保持计划足够内存（ONNX 建议 ≥1GB），或稍后重试。"
    },
    { status: 502 }
  );
}

/** 轻量唤醒 Render（仅 GET /health），可在分析前由前端或监控调用。 */
export async function GET() {
  const base = upstreamBase();
  if (!base) {
    return NextResponse.json(
      { ok: false, error: "未配置 INFER_SERVICE_URL，无法唤醒上游" },
      { status: 503 }
    );
  }
  try {
    const r = await fetch(`${base}/health`, {
      method: "GET",
      cache: "no-store",
      signal: AbortSignal.timeout(28_000)
    });
    const text = await r.text();
    return NextResponse.json(
      { ok: r.ok, status: r.status, upstream: base, body: r.ok ? safeJson(text) : text.slice(0, 500) },
      { status: r.ok ? 200 : 502 }
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ ok: false, error: msg, upstream: base }, { status: 502 });
  }
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text.slice(0, 500);
  }
}
