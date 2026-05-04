import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
/** Hobby 常为 60s；推理慢时可升级计划或拆分优化 */
export const maxDuration = 120;

function upstreamBase(): string {
  const raw =
    process.env.INFER_SERVICE_URL?.trim() ||
    process.env.NEXT_PUBLIC_INFER_SERVICE_URL?.trim() ||
    "";
  return raw.replace(/\/$/, "");
}

/** 浏览器同源调用本路由，服务端再 POST 至 Render；避免浏览器直连时的 CORS / 预检等问题。 */
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

  try {
    const upstream = await fetch(`${base}/detect`, {
      method: "POST",
      body: formData
    });

    const body = await upstream.arrayBuffer();
    const ct =
      upstream.headers.get("content-type")?.split(";")[0]?.trim() || "application/json";

    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "Content-Type": ct
      }
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json(
      { error: `转发推理服务失败: ${msg}`, upstreamBase: base },
      { status: 502 }
    );
  }
}
