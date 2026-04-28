# 可见光微裂缝检测（YOLO 检测框）

## 1) 准备数据

使用 YOLO 标注格式，推荐先做单类微裂缝检测。

- `images/train`, `images/val`, `images/test`
- `labels/train`, `labels/val`, `labels/test`
- `data.pvelad.yaml`（可复制 `data.pvelad.example.yaml` 改路径）

## 2) 训练

```bash
python train/crack_detector/train_crack_detector.py --data train/crack_detector/data.pvelad.yaml --model yolo11s.pt --imgsz 1280 --epochs 150 --batch 16 --device 0
```

## 3) 导出 ONNX 并放到前端模型目录

```bash
python train/crack_detector/export_to_onnx.py --weights runs/crack_detector/yolo11s-crack/weights/best.pt --imgsz 1280 --target public/model/visible-crack-detector.onnx
```

## 4) 前端接入建议

当前前端主流程用的是 `runYoloInference()`，可见光模型路径可替换为：

- `public/model/visible-crack-detector.onnx`

并把可见光类别映射改为：

- `["Crack"]`

