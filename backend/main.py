from __future__ import annotations

import json
import os
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from yolo_infer import run_yolo_inference, run_yolo_inference_with_debug

app = FastAPI(title="YOLO Web Inference")

INFER_ENGINE = os.environ.get("YOLOWEB_INFER_ENGINE", "onnx").strip().lower()

_cors_origins_raw = os.environ.get("YOLOWEB_CORS_ORIGINS", "").strip()
if _cors_origins_raw == "*":
    _origins = ["*"]
elif _cors_origins_raw:
    _origins = [x.strip() for x in _cors_origins_raw.split(",") if x.strip()]
else:
    # 未配置时放开任意来源（allow_credentials=False 下合法）。生产可自行改为显式域名列表。
    _origins = ["*"]

# 本推理 API 不使用 Cookie / Authorization；allow_credentials=False 可与 allow_origins="*" 并存，避免浏览器 CORS 静默失败。
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

_DEFAULT_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "model"))

VISIBLE_MODEL = os.environ.get("YOLOWEB_VISIBLE_ONNX") or os.path.join(_DEFAULT_REPO, "yolov11s-pv.onnx")
THERMAL_MODEL = os.environ.get("YOLOWEB_THERMAL_ONNX") or os.path.join(_DEFAULT_REPO, "thermal-hotspot.onnx")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "inferEngine": INFER_ENGINE}


@app.get("/health/models")
def health_models() -> dict[str, str | bool]:
    """检查默认或环境变量指向的 ONNX 是否在磁盘存在（Deploy 后用此路由排查路径问题）。"""
    return {
        "visiblePath": VISIBLE_MODEL,
        "thermalPath": THERMAL_MODEL,
        "visibleExists": os.path.isfile(VISIBLE_MODEL),
        "thermalExists": os.path.isfile(THERMAL_MODEL),
    }


@app.post("/detect")
def detect(
    image: Annotated[bytes, File(description="JPEG/PNG 图像")],
    model: Annotated[str, Form(description="visible 或 thermal")],
    *,
    debug: Annotated[str, Form()] = "false",
    options_json: Annotated[str | None, Form(alias="options")] = None,
) -> dict:
    mk = model.strip().lower()
    path = {"visible": VISIBLE_MODEL, "thermal": THERMAL_MODEL}.get(mk)
    if not path:
        raise HTTPException(status_code=400, detail="model 必须为 visible 或 thermal")
    if not os.path.isfile(path):
        raise HTTPException(status_code=500, detail=f"模型文件不存在: {path}")

    opts = {}
    if options_json:
        try:
            parsed = json.loads(options_json)
            if isinstance(parsed, dict):
                opts = parsed
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"options JSON 无效: {e}") from e

    raw_cn = opts.get("classNames")
    if raw_cn is not None and not isinstance(raw_cn, list):
        raise HTTPException(status_code=400, detail="classNames 必须为字符串数组")

    if mk == "thermal":
        cn = raw_cn if raw_cn else ["Hotspot"]
    else:
        cn = raw_cn if raw_cn else ["Clean", "Dust", "Bird", "Electrical", "Physical", "Snow"]

    ho = opts.get("hasObjectness", False)
    if ho == "auto":
        resolved_ho = "auto"
    else:
        resolved_ho = bool(ho)

    bf_raw = opts.get("boxFormat", "cxcywh")
    bf = bf_raw if bf_raw in {"cxcywh", "xyxy", "auto"} else "cxcywh"

    fb_raw = opts.get("fallbackToAlternateHead")
    if fb_raw is None:
        fb_ok = mk == "visible"
    else:
        fb_ok = bool(fb_raw)

    dbg = debug.strip().lower() in {"1", "true", "yes", "on"}
    infer_kw = dict(
        class_names=cn,
        has_objectness=resolved_ho,
        apply_sigmoid=bool(opts.get("applySigmoid", False)),
        fallback_to_alternate_head=fb_ok,
        min_box_size=float(opts.get("minBoxSize", 0 if mk == "thermal" else 2)),
        max_box_area_ratio=float(opts.get("maxBoxAreaRatio", 1)),
        score_threshold=float(opts.get("scoreThreshold", 0.65)),
        iou_threshold=float(opts.get("iouThreshold", 0.5)),
        max_detections=int(opts.get("maxDetections", 100 if mk == "thermal" else 120)),
        box_format=bf,
    )

    try:
        if INFER_ENGINE in ("ultra", "ultralytics"):
            from yolo_ultralytics_infer import (  # noqa: PLC0415
                ultralytics_debug_stub,
                ultralytics_infer,
            )

            isz_raw = opts.get("imgsz")
            imgsz: int | None
            if isinstance(isz_raw, (int, float)):
                imgsz = int(isz_raw)
            else:
                imgsz = None

            boxes = ultralytics_infer(
                image,
                path,
                conf=float(infer_kw["score_threshold"]),
                iou=float(infer_kw["iou_threshold"]),
                max_detections=int(infer_kw["max_detections"]),
                imgsz=imgsz,
            )
            if dbg:
                return {"boxes": boxes, "debug": ultralytics_debug_stub(boxes)}
            return {"boxes": boxes}

        if dbg:
            boxes, dbg_info = run_yolo_inference_with_debug(image, path, **infer_kw)
            return {"boxes": boxes, "debug": dbg_info}

        boxes = run_yolo_inference(image, path, **infer_kw)
        return {"boxes": boxes}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"推理失败: {type(e).__name__}: {e}") from e
