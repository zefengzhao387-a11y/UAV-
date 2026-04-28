"""
微裂缝检测模型训练脚本（YOLO11）。

使用方式：
python train/crack_detector/train_crack_detector.py --data train/crack_detector/data.pvelad.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train crack detector with YOLO11")
  parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="YOLO 数据集配置文件路径（yaml）"
  )
  parser.add_argument("--model", type=str, default="yolo11s.pt", help="预训练基座模型")
  parser.add_argument("--imgsz", type=int, default=1280, help="训练输入分辨率（微裂缝建议 >=1024）")
  parser.add_argument("--epochs", type=int, default=150, help="训练轮数")
  parser.add_argument("--batch", type=int, default=16, help="批大小")
  parser.add_argument("--device", type=str, default="0", help="训练设备，例如 0 / cpu")
  parser.add_argument("--project", type=str, default="runs/crack_detector", help="训练输出目录")
  parser.add_argument("--name", type=str, default="yolo11s-crack", help="本次实验名")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  data_path = Path(args.data)
  if not data_path.exists():
    raise FileNotFoundError(f"数据配置文件不存在: {data_path}")

  model = YOLO(args.model)
  model.train(
    data=str(data_path),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    project=args.project,
    name=args.name,
    workers=8,
    close_mosaic=10,
    mosaic=1.0,
    degrees=0.0,
    fliplr=0.5
  )


if __name__ == "__main__":
  main()
