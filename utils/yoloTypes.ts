/**
 * 与后端 `POST /detect` 请求体字段（options JSON）及响应框结构对齐。
 */

export interface DetectionBox {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
  score: number;
  classId: number;
  className: string;
}

export interface PostprocessDebugInfo {
  outputDims: number[];
  featureLen: number;
  numBoxes: number;
  coordinateScale: number;
  effectiveHasObjectness: boolean;
  effectiveBoxFormat: "cxcywh" | "xyxy";
  classStart: number;
  numClasses: number;
  returnedBoxes: number;
  topBoxes: Array<{ className: string; score: number; classId: number }>;
  fallbackUsed?: boolean;
  fallbackTried?: Array<{
    hasObjectness: boolean;
    boxFormat: "cxcywh" | "xyxy";
    returnedBoxes: number;
    scoreThresholdUsed: number;
  }>;
}

export interface AnalyzeOptions {
  modelPath?: string;
  classNames?: string[];
  hasObjectness?: boolean | "auto";
  applySigmoid?: boolean;
  fallbackToAlternateHead?: boolean;
  minBoxSize?: number;
  maxBoxAreaRatio?: number;
  scoreThreshold?: number;
  iouThreshold?: number;
  maxDetections?: number;
  boxFormat?: "cxcywh" | "xyxy" | "auto";
}
