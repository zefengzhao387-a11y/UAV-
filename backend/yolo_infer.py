"""
ONNX YOLO 推理：预处理与 postprocess 与前端 modelHelper.ts 对齐（cover + 居中贴图到 640）。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_INPUT_SIZE = 640


@dataclass
class PreprocessMeta:
    scale: float
    offset_x: float
    offset_y: float
    src_width: float
    src_height: float
    input_size: int


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def preprocess_image(rgb: Image.Image, input_size: int = MODEL_INPUT_SIZE) -> tuple[np.ndarray, PreprocessMeta]:
    src_w = float(rgb.size[0])
    src_h = float(rgb.size[1])
    scale = max(input_size / src_w, input_size / src_h)
    resized_w = src_w * scale
    resized_h = src_h * scale
    offset_x = (input_size - resized_w) / 2
    offset_y = (input_size - resized_h) / 2

    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    resized = rgb.resize((int(round(resized_w)), int(round(resized_h))), Image.Resampling.BILINEAR)
    paste_x = int(round(offset_x))
    paste_y = int(round(offset_y))
    canvas.paste(resized, (paste_x, paste_y))

    arr = np.asarray(canvas).astype(np.float32) / 255.0
    # HWC RGB -> NCHW
    chw = np.transpose(arr, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0)
    meta = PreprocessMeta(
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        src_width=src_w,
        src_height=src_h,
        input_size=input_size,
    )
    return tensor, meta


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def intersection_over_union(a: dict, b: dict) -> float:
    xa = max(a["x_min"], b["x_min"])
    ya = max(a["y_min"], b["y_min"])
    xb = min(a["x_max"], b["x_max"])
    yb = min(a["y_max"], b["y_max"])

    iw = max(0.0, xb - xa)
    ih = max(0.0, yb - ya)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a["x_max"] - a["x_min"]) * max(0.0, a["y_max"] - a["y_min"])
    bb = max(0.0, b["x_max"] - b["x_min"]) * max(0.0, b["y_max"] - b["y_min"])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def remap_to_original(box: dict, meta: PreprocessMeta) -> dict:
    return {
        **box,
        "x_min": clamp((box["x_min"] - meta.offset_x) / meta.scale, 0.0, meta.src_width),
        "y_min": clamp((box["y_min"] - meta.offset_y) / meta.scale, 0.0, meta.src_height),
        "x_max": clamp((box["x_max"] - meta.offset_x) / meta.scale, 0.0, meta.src_width),
        "y_max": clamp((box["y_max"] - meta.offset_y) / meta.scale, 0.0, meta.src_height),
    }


def _read_raw(raw: np.ndarray, d1: int, d2: int, box_index: int, feature: int, features_on_d1: bool) -> float:
    if features_on_d1:
        return float(raw[feature, box_index])
    return float(raw[box_index, feature])


def postprocess_detections(
    output: np.ndarray,
    meta: PreprocessMeta,
    class_names: list[str],
    has_objectness: bool | Literal["auto"],
    apply_sigmoid: bool,
    min_box_size: float,
    max_box_area_ratio: float,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
    box_format: Literal["cxcywh", "xyxy", "auto"],
) -> list[dict[str, Any]]:
    if output.ndim != 3:
        raise ValueError(f"期望 3 维模型输出，实际 {output.ndim}")
    dims = output.shape
    _, d1, d2 = dims
    feature_len = min(d1, d2)
    num_boxes = max(d1, d2)
    features_on_d1 = d1 == feature_len
    flat = np.squeeze(output, axis=0)
    raw = flat if flat.shape == (d1, d2) else output.reshape(d1, d2)

    if feature_len < 5:
        raise ValueError(f"特征维度异常：{feature_len}")

    coordinate_scale = 1.0

    def pick_has_objectness() -> bool:
        if has_objectness != "auto":
            return bool(has_objectness)
        expected = len(class_names)
        noc = feature_len - 4
        woc = feature_len - 5
        diff_no = abs(noc - expected)
        diff_wo = abs(woc - expected)
        if noc == expected and woc != expected:
            return False
        if woc == expected and noc != expected:
            return True
        return diff_wo <= diff_no

    effective_has_objectness = pick_has_objectness()
    class_start = 5 if effective_has_objectness else 4
    num_classes = feature_len - class_start
    if num_classes <= 0:
        raise ValueError(f"类别维度异常 feature_len={feature_len}")

    def run_once(fmt: Literal["cxcywh", "xyxy"]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for i in range(num_boxes):
            cx = _read_raw(raw, d1, d2, i, 0, features_on_d1) * coordinate_scale
            cy = _read_raw(raw, d1, d2, i, 1, features_on_d1) * coordinate_scale
            w = _read_raw(raw, d1, d2, i, 2, features_on_d1) * coordinate_scale
            h = _read_raw(raw, d1, d2, i, 3, features_on_d1) * coordinate_scale

            if fmt == "cxcywh":
                if w <= 0 or h <= 0:
                    continue
                x_min = cx - w / 2
                y_min = cy - h / 2
                x_max = cx + w / 2
                y_max = cy + h / 2
            else:
                if w <= cx or h <= cy:
                    continue
                x_min = cx
                y_min = cy
                x_max = w
                y_max = h

            best_class_id = 0
            best_score = float("-inf")
            for c in range(num_classes):
                raw_cls = _read_raw(raw, d1, d2, i, class_start + c, features_on_d1)
                cls_score = sigmoid(raw_cls) if apply_sigmoid else raw_cls
                if cls_score > best_score:
                    best_score = cls_score
                    best_class_id = c

            if effective_has_objectness:
                raw_obj = _read_raw(raw, d1, d2, i, 4, features_on_d1)
                obj_score = sigmoid(raw_obj) if apply_sigmoid else raw_obj
                best_score *= obj_score

            if not math.isfinite(best_score) or best_score < score_threshold:
                continue
            if x_max <= x_min or y_max <= y_min:
                continue

            raw_box = {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "score": best_score,
                "class_id": best_class_id,
                "class_name": class_names[best_class_id] if best_class_id < len(class_names) else f"Class-{best_class_id}",
            }
            mapped = remap_to_original(raw_box, meta)
            mw = mapped["x_max"] - mapped["x_min"]
            mh = mapped["y_max"] - mapped["y_min"]
            if mw < min_box_size or mh < min_box_size:
                continue
            area_ratio = (mw * mh) / (meta.src_width * meta.src_height)
            if area_ratio > max_box_area_ratio:
                continue
            if mapped["x_max"] <= mapped["x_min"] or mapped["y_max"] <= mapped["y_min"]:
                continue
            candidates.append(mapped)

        candidates.sort(key=lambda b: b["score"], reverse=True)
        selected: list[dict[str, Any]] = []
        while candidates:
            current = candidates.pop(0)
            selected.append(current)
            remain: list[dict[str, Any]] = []
            for b in candidates:
                if intersection_over_union(current, b) <= iou_threshold:
                    remain.append(b)
            candidates = remain
        return selected[:max_detections]

    if box_format == "auto":
        cx_boxes = run_once("cxcywh")
        xy_boxes = run_once("xyxy")

        def total_area_ratio(boxes: list[dict[str, Any]]) -> float:
            s = 0.0
            for b in boxes:
                ww = max(0.0, b["x_max"] - b["x_min"])
                hh = max(0.0, b["y_max"] - b["y_min"])
                s += (ww * hh) / (meta.src_width * meta.src_height)
            return s

        cx_score = total_area_ratio(cx_boxes) - len(cx_boxes) * 1e-5
        xy_score = total_area_ratio(xy_boxes) - len(xy_boxes) * 1e-5
        return xy_boxes if xy_score > cx_score else cx_boxes

    return run_once(box_format)


def serialize_box(b: dict[str, Any]) -> dict[str, Any]:
    return {
        "xMin": float(b["x_min"]),
        "yMin": float(b["y_min"]),
        "xMax": float(b["x_max"]),
        "yMax": float(b["y_max"]),
        "score": float(b["score"]),
        "classId": int(b["class_id"]),
        "className": str(b["class_name"]),
    }


_SESSIONS: dict[str, ort.InferenceSession] = {}


def get_session(model_path: str) -> ort.InferenceSession:
    if model_path not in _SESSIONS:
        _SESSIONS[model_path] = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return _SESSIONS[model_path]


def run_yolo_inference(
    image_bytes: bytes,
    model_path: str,
    *,
    class_names: list[str],
    has_objectness: bool | str = False,
    apply_sigmoid: bool = False,
    fallback_to_alternate_head: bool = True,
    min_box_size: float = 6.0,
    max_box_area_ratio: float = 1.0,
    score_threshold: float = 0.65,
    iou_threshold: float = 0.5,
    max_detections: int = 80,
    box_format: Literal["cxcywh", "xyxy", "auto"] = "cxcywh",
) -> list[dict[str, Any]]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor, meta = preprocess_image(img)

    sess = get_session(model_path)
    inp = sess.get_inputs()[0].name
    outp = sess.get_outputs()[0].name
    out = sess.run([outp], {inp: tensor})[0]

    ho = "auto" if has_objectness == "auto" else bool(has_objectness)

    primary = postprocess_detections(
        out,
        meta,
        class_names,
        ho,
        apply_sigmoid,
        min_box_size,
        max_box_area_ratio,
        score_threshold,
        iou_threshold,
        max_detections,
        box_format,
    )

    if fallback_to_alternate_head and len(primary) == 0:
        fb_thr = max(0.25, score_threshold * 0.75)
        if ho == "auto":
            tries = [False, True]
        else:
            tries = [not ho]
        for obj in tries:
            fb = postprocess_detections(
                out,
                meta,
                class_names,
                obj,
                apply_sigmoid,
                min_box_size,
                max_box_area_ratio,
                fb_thr,
                iou_threshold,
                max_detections,
                box_format,
            )
            if len(fb) > 0:
                return [serialize_box(b) for b in fb]

    return [serialize_box(b) for b in primary]


def debug_info_from_boxes(
    out: np.ndarray,
    meta: PreprocessMeta,
    final_boxes: list[dict[str, Any]],
    *,
    effective_has_objectness: bool,
    effective_box_format: str,
    class_start: int,
    fallback_used: bool,
    fallback_tried: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if out.ndim != 3:
        return {}
    _, d1, d2 = out.shape
    feature_len = min(d1, d2)
    num_boxes = max(d1, d2)
    num_classes = feature_len - class_start
    top = sorted(final_boxes, key=lambda b: b["score"], reverse=True)[:5]
    top_boxes = [{"className": b["className"], "score": b["score"], "classId": b["classId"]} for b in top]

    return {
        "outputDims": [int(x) for x in out.shape],
        "featureLen": feature_len,
        "numBoxes": num_boxes,
        "coordinateScale": 1,
        "effectiveHasObjectness": effective_has_objectness,
        "effectiveBoxFormat": effective_box_format,
        "classStart": class_start,
        "numClasses": num_classes,
        "returnedBoxes": len(final_boxes),
        "topBoxes": top_boxes,
        "fallbackUsed": fallback_used,
        "fallbackTried": fallback_tried,
    }


def run_yolo_inference_with_debug(
    image_bytes: bytes,
    model_path: str,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """与前端 debug 面板字段接近；内部复跑一次推理以获得 output tensor。"""
    class_names = kwargs["class_names"]
    apply_sigmoid = kwargs.get("apply_sigmoid", False)
    min_box_size = kwargs.get("min_box_size", 6.0)
    max_box_area_ratio = kwargs.get("max_box_area_ratio", 1.0)
    score_threshold = kwargs.get("score_threshold", 0.65)
    iou_threshold = kwargs.get("iou_threshold", 0.5)
    max_detections = kwargs.get("max_detections", 80)
    box_format_opt = kwargs.get("box_format", "cxcywh")
    fallback_to_alternate_head = kwargs.get("fallback_to_alternate_head", True)
    ho_opt = kwargs.get("has_objectness", False)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor, meta = preprocess_image(img)

    sess = get_session(model_path)
    inp = sess.get_inputs()[0].name
    outp = sess.get_outputs()[0].name
    out = sess.run([outp], {inp: tensor})[0]

    if out.ndim != 3:
        raise ValueError("模型输出维度异常")
    _, d1, d2 = out.shape
    feature_len = min(d1, d2)

    flat = np.squeeze(out, axis=0)
    raw_arr = flat if flat.shape == (d1, d2) else out.reshape(d1, d2)

    def pick_has_objectness(h_in: bool | str) -> bool:
        if h_in != "auto":
            return bool(h_in)
        expected = len(class_names)
        noc = feature_len - 4
        woc = feature_len - 5
        diff_no = abs(noc - expected)
        diff_wo = abs(woc - expected)
        if noc == expected and woc != expected:
            return False
        if woc == expected and noc != expected:
            return True
        return diff_wo <= diff_no

    def total_area_ratio(boxes: list[dict[str, Any]]) -> float:
        s = 0.0
        for b in boxes:
            ww = max(0.0, b["x_max"] - b["x_min"])
            hh = max(0.0, b["y_max"] - b["y_min"])
            s += (ww * hh) / (meta.src_width * meta.src_height)
        return s

    def choose_format(h_obj: bool, thr: float) -> tuple[str, list[dict[str, Any]]]:
        fmt = box_format_opt
        if fmt != "auto":
            bf: Literal["cxcywh", "xyxy"] = "cxcywh" if fmt == "cxcywh" else "xyxy"
            boxes_int = postprocess_detections(
                out,
                meta,
                list(class_names),
                h_obj,
                apply_sigmoid,
                float(min_box_size),
                float(max_box_area_ratio),
                float(thr),
                float(iou_threshold),
                int(max_detections),
                bf,
            )
            return bf, boxes_int

        cx_b = postprocess_detections(
            out,
            meta,
            list(class_names),
            h_obj,
            apply_sigmoid,
            float(min_box_size),
            float(max_box_area_ratio),
            float(thr),
            float(iou_threshold),
            int(max_detections),
            "cxcywh",
        )
        xy_b = postprocess_detections(
            out,
            meta,
            list(class_names),
            h_obj,
            apply_sigmoid,
            float(min_box_size),
            float(max_box_area_ratio),
            float(thr),
            float(iou_threshold),
            int(max_detections),
            "xyxy",
        )
        cx_sc = total_area_ratio(cx_b) - len(cx_b) * 1e-5
        xy_sc = total_area_ratio(xy_b) - len(xy_b) * 1e-5
        if xy_sc > cx_sc:
            return "xyxy", xy_b
        return "cxcywh", cx_b

    effective_primary_has_objectness = pick_has_objectness(ho_opt)

    primary_fmt, primary_boxes_internal = choose_format(effective_primary_has_objectness, score_threshold)

    used_fallback = False
    fb_tried: list[dict[str, Any]] | None = None
    final_boxes_int = primary_boxes_internal
    final_has_objectness = effective_primary_has_objectness
    final_box_format = primary_fmt

    if fallback_to_alternate_head and len(primary_boxes_internal) == 0:
        used_fallback = True
        fb_tried = []
        fb_thr = max(0.25, score_threshold * 0.75)
        if ho_opt == "auto":
            try_orders = [False, True]
        else:
            try_orders = [not bool(ho_opt)]
        for obj in try_orders:
            fmt_try, bx = choose_format(obj, fb_thr)
            fb_tried.append(
                {
                    "hasObjectness": obj,
                    "boxFormat": fmt_try,
                    "returnedBoxes": len(bx),
                    "scoreThresholdUsed": fb_thr,
                }
            )
            if len(bx) > 0:
                final_has_objectness = obj
                final_box_format = fmt_try
                final_boxes_int = bx
                break

    class_start = 5 if final_has_objectness else 4
    serialized = [serialize_box(b) for b in final_boxes_int]
    dbg = debug_info_from_boxes(
        out,
        meta,
        serialized,
        effective_has_objectness=final_has_objectness,
        effective_box_format=final_box_format,
        class_start=class_start,
        fallback_used=used_fallback,
        fallback_tried=fb_tried,
    )
    return serialized, dbg
