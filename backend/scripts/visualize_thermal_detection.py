#!/usr/bin/env python3
"""
读取本地热力图 → （可选 POST Render /detect 或载入 JSON）→ 在原图上绘制检测框保存。

示例：
  python backend/scripts/visualize_thermal_detection.py --image thermal.png \\
    --out public/thermal_detection_preview.png

远端 502 多时可稍后重试，或用已有 JSON：
  python backend/scripts/visualize_thermal_detection.py --image thermal.png \\
    --boxes-json backend/scripts/sample_thermal_boxes.json --out preview.png

"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
import urllib.request
import urllib.error

from PIL import Image, ImageDraw, ImageFont


def build_multipart(image_bytes: bytes, filename: str, options: dict) -> tuple[bytes, str]:
    opts = json.dumps(options, ensure_ascii=False)
    bnd = uuid.uuid4().hex
    CRLF = b"\r\n"

    def part(name: str, value: str) -> bytes:
        return (
            b"--" + bnd.encode()
            + CRLF
            + f'Content-Disposition: form-data; name="{name}"'.encode()
            + CRLF
            + CRLF
            + value.encode()
            + CRLF
        )

    body = (
        part("model", "thermal")
        + part("debug", "false")
        + part("options", opts)
        + b"--"
        + bnd.encode()
        + CRLF
        + f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode()
        + CRLF
        + b"Content-Type: image/png"
        + CRLF
        + CRLF
        + image_bytes
        + CRLF
        + b"--"
        + bnd.encode()
        + b"--"
        + CRLF
    )
    ct = f"multipart/form-data; boundary={bnd}"
    return body, ct


def fetch_boxes(api_base: str, image_bytes: bytes, filename: str, retries: int) -> list[dict]:
    options = {
        # 与你的热力导出一致时请补全两段名字；单列时仍可检出，占位名见 Class-N
        "classNames": ["background", "Hotspot"],
        "hasObjectness": False,
        "applySigmoid": False,
        "fallbackToAlternateHead": False,
        "minBoxSize": 0,
        "maxBoxAreaRatio": 1,
        "scoreThreshold": 0.2,
        "iouThreshold": 0.45,
        "maxDetections": 100,
        "boxFormat": "cxcywh",
    }
    body, ct = build_multipart(image_bytes, filename, options)
    url = f"{api_base.rstrip('/')}/detect"
    last_err: Exception | None = None

    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": ct}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode())
                boxes = data.get("boxes") or []
                if not isinstance(boxes, list):
                    raise SystemExit("响应里没有 boxes 列表")
                return boxes
        except urllib.error.HTTPError as e:
            body_frag = e.read().decode(errors="ignore")[:500]
            if e.code in (502, 503, 504) and attempt < retries - 1:
                wait = min(45, 6 * (attempt + 1))
                print(f"HTTP {e.code}，{wait}s 后重试 ({attempt + 1}/{retries})…", flush=True)
                time.sleep(wait)
                last_err = RuntimeError(f"HTTP {e.code}: {body_frag}")
                continue
            raise SystemExit(f"HTTP {e.code}: {body_frag}") from e
        except OSError as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(10)
                continue
            raise

    raise SystemExit(f"请求失败（已重试 {retries} 次）: {last_err}")


def draw_boxes(im: Image.Image, boxes: list[dict]) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, min(im.size) // 60))
    except OSError:
        font = ImageFont.load_default()

    for b in boxes:
        x1, y1 = float(b["xMin"]), float(b["yMin"])
        x2, y2 = float(b["xMax"]), float(b["yMax"])
        label = f"{b.get('className', '?')} {float(b['score']):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:4]
        draw.rectangle([x1, max(0, y1 - th - 6), x1 + tw + 8, y1], fill=(255, 60, 60))
        draw.text((x1 + 4, max(0, y1 - th - 4)), label, fill=(255, 255, 255), font=font)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="热力检测可视化（远程 API）")
    ap.add_argument(
        "--image",
        type=Path,
        required=True,
        help="热力图 PNG/JPG",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("public/thermal_detection_preview.png"),
        help="输出带框图片路径",
    )
    ap.add_argument(
        "--api-base",
        default="https://uav.onrender.com",
        help="推理服务根 URL（--boxes-json 时忽略）",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=5,
        help="/detect 在 502/503/504 时的重试次数",
    )
    ap.add_argument(
        "--boxes-json",
        type=Path,
        default=None,
        help="跳过网络，直接使用 detect 接口同结构的 JSON（含 boxes 数组）",
    )
    args = ap.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"找不到图像: {args.image}")

    img = Image.open(args.image).convert("RGB")

    if args.boxes_json:
        blob = json.loads(args.boxes_json.read_text(encoding="utf-8"))
        boxes = blob.get("boxes")
        if not isinstance(boxes, list):
            raise SystemExit("--boxes-json 需提供含 boxes 列表的 JSON")
    else:
        raw = args.image.read_bytes()
        boxes = fetch_boxes(args.api_base, raw, args.image.name, args.retries)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    vis = draw_boxes(img, boxes)
    vis.save(args.out, optimize=True)

    print(f"检出 {len(boxes)} 个框，已保存: {args.out.resolve()}")


if __name__ == "__main__":
    main()
