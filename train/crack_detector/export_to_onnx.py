"""
将训练好的微裂缝检测权重导出为 ONNX，并复制到前端模型目录。

使用方式：
python train/crack_detector/export_to_onnx.py --weights runs/crack_detector/yolo11s-crack/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Export crack detector weights to ONNX")
  parser.add_argument("--weights", type=str, required=True, help="best.pt 权重路径")
  parser.add_argument("--imgsz", type=int, default=1280, help="导出输入尺寸，需与训练策略一致")
  parser.add_argument(
    "--target",
    type=str,
    default="public/model/visible-crack-detector.onnx",
    help="导出后复制到前端的目标路径"
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  weights_path = Path(args.weights)
  if not weights_path.exists():
    raise FileNotFoundError(f"权重文件不存在: {weights_path}")

  model = YOLO(str(weights_path))
  result_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True))

  target_path = Path(args.target)
  target_path.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(result_path, target_path)
  print(f"ONNX 已复制到: {target_path.resolve()}")


if __name__ == "__main__":
  main()
