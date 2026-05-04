"""
使用 Ultralytics 内置 predict（与 YOLO 导出模型自带的 letterbox / 缩放等流水线一致）。
不经过本项目中手写的 ONNX 手工预处理与解码。
"""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image

_cached: dict[str, Any] = {}


def _model(path: str):
    from ultralytics import YOLO  # noqa: PLC0415

    if path not in _cached:
        _cached[path] = YOLO(path)
    return _cached[path]


def _class_name(names: dict, cid: int) -> str:
    if cid in names:
        return str(names[cid])
    if str(cid) in names:
        return str(names[str(cid)])
    return f"Class-{cid}"


def ultralytics_infer(
    image_bytes: bytes,
    model_path: str,
    *,
    conf: float,
    iou: float,
    max_detections: int,
    imgsz: int | None,
) -> list[dict[str, Any]]:
    m = _model(model_path)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    isz = int(imgsz) if imgsz else int(os.environ.get("YOLOWEB_IMGSZ", "640"))

    results = m.predict(
        source=img,
        conf=float(conf),
        iou=float(iou),
        max_det=max(1, int(max_detections)),
        imgsz=isz,
        verbose=False,
    )

    nm = getattr(m, "names", None)
    names_dict: dict = nm if isinstance(nm, dict) else {}

    boxes_out: list[dict[str, Any]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            cid = int(cls_ids[i])
            boxes_out.append(
                {
                    "xMin": float(xyxy[i][0]),
                    "yMin": float(xyxy[i][1]),
                    "xMax": float(xyxy[i][2]),
                    "yMax": float(xyxy[i][3]),
                    "score": float(scores[i]),
                    "classId": cid,
                    "className": _class_name(names_dict, cid),
                }
            )
    return boxes_out


def ultralytics_debug_stub(boxes: list[dict[str, Any]]) -> dict[str, Any]:
    top = sorted(boxes, key=lambda b: b["score"], reverse=True)[:5]
    return {
        "outputDims": [-1],
        "featureLen": 0,
        "numBoxes": len(boxes),
        "coordinateScale": 1.0,
        "effectiveHasObjectness": False,
        "effectiveBoxFormat": "xyxy",
        "classStart": 0,
        "numClasses": 0,
        "returnedBoxes": len(boxes),
        "topBoxes": [{"className": b["className"], "score": b["score"], "classId": b["classId"]} for b in top],
        "fallbackUsed": False,
        "engine": "ultralytics",
    }
