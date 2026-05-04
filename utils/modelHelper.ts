"use client";

import type { DetectionBox } from "@/utils/yoloTypes";

export type { AnalyzeOptions, DetectionBox, PostprocessDebugInfo } from "@/utils/yoloTypes";

/**
 * 在叠加 Canvas 上绘制检测框与标签。
 */
export function drawBoundingBoxes(
  canvas: HTMLCanvasElement,
  imageElement: HTMLImageElement,
  boxes: DetectionBox[]
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const width = imageElement.clientWidth;
  const height = imageElement.clientHeight;
  const naturalWidth = imageElement.naturalWidth || width;
  const naturalHeight = imageElement.naturalHeight || height;

  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);

  const containScale = Math.min(width / naturalWidth, height / naturalHeight);
  const renderedWidth = naturalWidth * containScale;
  const renderedHeight = naturalHeight * containScale;
  const offsetX = (width - renderedWidth) / 2;
  const offsetY = (height - renderedHeight) / 2;

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ff4d4f";
  ctx.font = "12px sans-serif";
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const x = offsetX + box.xMin * containScale;
    const y = offsetY + box.yMin * containScale;
    const w = (box.xMax - box.xMin) * containScale;
    const h = (box.yMax - box.yMin) * containScale;
    const label = `${box.className}: ${(box.score * 100).toFixed(0)}%`;

    ctx.strokeRect(x, y, w, h);

    const textWidth = ctx.measureText(label).width;
    const labelHeight = 18;
    const labelY = Math.max(0, y - labelHeight - 2);
    ctx.fillStyle = "rgba(255, 77, 79, 0.75)";
    ctx.fillRect(x, labelY, textWidth + 10, labelHeight);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 5, labelY + 3);
  });
}
