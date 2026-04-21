"""
Pure ONNX Runtime YOLO Detector
================================
Zero-dependency replacement for ultralytics.YOLO — runs on any ARM64/x86
device with only onnxruntime + numpy + opencv (no PyTorch required).

Designed for Jetson Orin Nano / Raspberry Pi edge deployment.
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Any, Tuple


class ONNXDetector:
    """
    Lightweight YOLOv8 ONNX inference engine.

    Replaces `from ultralytics import YOLO` with pure onnxruntime,
    eliminating the entire PyTorch dependency tree.
    """

    def __init__(self, model_path: str, imgsz: int = 640):
        """
        Args:
            model_path: Path to the .onnx model file.
            imgsz: Input image size for the model (square).
        """
        self.imgsz = imgsz

        # Pick the best available execution provider
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        providers.append("CPUExecutionProvider")

        print(f"[ONNX] Loading {model_path} …")
        print(f"[ONNX] Available providers: {available}")
        print(f"[ONNX] Using providers: {providers}")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Probe output shape to detect num_classes
        # YOLOv8 output: [1, 4+num_classes, num_predictions]
        out_shape = self.session.get_outputs()[0].shape
        print(f"[ONNX] Model input: {self.session.get_inputs()[0].shape}")
        print(f"[ONNX] Model output: {out_shape}")
        print(f"[ONNX] Ready.")

    # ------------------------------------------------------------------
    # Public API  (drop-in replacement for how video_pipeline.py uses YOLO)
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray,
               conf: float = 0.5,
               iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR numpy frame.

        Args:
            frame: Input image (BGR, HWC, uint8).
            conf: Minimum confidence threshold.
            iou_thresh: IoU threshold for NMS.

        Returns:
            List of dicts with 'bbox' [x1,y1,x2,y2] and 'confidence'.
        """
        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocess: letterbox resize → normalize → NCHW
        blob, ratio, (pad_w, pad_h) = self._preprocess(frame)

        # 2. Inference
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        preds = outputs[0]  # shape: [1, 4+nc, N] or [1, N, 4+nc]

        # 3. Postprocess: decode boxes, filter, NMS, scale back
        detections = self._postprocess(
            preds, conf, iou_thresh, ratio, pad_w, pad_h, orig_w, orig_h)

        return detections

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox resize + normalize to [0,1] + HWC→NCHW.
        Returns (blob, ratio, (pad_w, pad_h)).
        """
        img_h, img_w = frame.shape[:2]
        target = self.imgsz

        # Compute scale to fit the longer side into `target`
        ratio = min(target / img_w, target / img_h)
        new_w = int(round(img_w * ratio))
        new_h = int(round(img_h * ratio))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square (target × target) with gray (114)
        pad_w = (target - new_w) // 2
        pad_h = (target - new_h) // 2
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Normalize to [0, 1], convert BGR → RGB, HWC → CHW → NCHW
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # CHW
        blob = np.expand_dims(blob, axis=0)     # NCHW

        return blob, ratio, (pad_w, pad_h)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(self, preds: np.ndarray,
                     conf_thresh: float, iou_thresh: float,
                     ratio: float, pad_w: int, pad_h: int,
                     orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        """
        Decode YOLOv8 ONNX output. Handles two export formats:
          - End-to-end NMS: [1, N, 6]  =  x1, y1, x2, y2, score, class_id
          - Raw predictions: [1, 4+nc, N]  =  cx, cy, w, h, class_scores...
        """
        preds = preds[0]  # squeeze batch dim

        # -- Format A: End-to-end NMS export [N, 6] --
        if preds.ndim == 2 and preds.shape[1] == 6:
            results = []
            for row in preds:
                x1_r, y1_r, x2_r, y2_r, score, cls_id = row
                if score < conf_thresh:
                    continue
                x1_s = (float(x1_r) - pad_w) / ratio
                y1_s = (float(y1_r) - pad_h) / ratio
                x2_s = (float(x2_r) - pad_w) / ratio
                y2_s = (float(y2_r) - pad_h) / ratio
                x1_s = max(0, min(x1_s, orig_w))
                y1_s = max(0, min(y1_s, orig_h))
                x2_s = max(0, min(x2_s, orig_w))
                y2_s = max(0, min(y2_s, orig_h))
                if x2_s - x1_s < 2 or y2_s - y1_s < 2:
                    continue
                results.append({
                    "bbox": [int(x1_s), int(y1_s), int(x2_s), int(y2_s)],
                    "confidence": float(score),
                })
            return results

        # -- Format B: Raw predictions [4+nc, N] --
        if preds.shape[0] > preds.shape[1]:
            preds = preds.T

        boxes = preds[:4, :]
        scores = preds[4:, :]

        if scores.shape[0] > 1:
            class_scores = np.max(scores, axis=0)
        else:
            class_scores = scores[0]

        mask = class_scores >= conf_thresh
        if not np.any(mask):
            return []

        boxes = boxes[:, mask]
        class_scores = class_scores[mask]

        cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
        x1 = (cx - w / 2 - pad_w) / ratio
        y1 = (cy - h / 2 - pad_h) / ratio
        x2 = (cx + w / 2 - pad_w) / ratio
        y2 = (cy + h / 2 - pad_h) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        dets = np.stack([x1, y1, x2, y2, class_scores], axis=1)
        keep_indices = self._nms_numpy(dets, iou_thresh)
        dets = dets[keep_indices]

        results = []
        for det in dets:
            results.append({
                "bbox": [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
                "confidence": float(det[4]),
            })
        return results

    # ------------------------------------------------------------------
    # NMS (pure numpy, no torchvision dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _nms_numpy(dets: np.ndarray, iou_thresh: float) -> List[int]:
        """
        Non-Maximum Suppression on [M, 5] array (x1, y1, x2, y2, score).
        Returns indices to keep.
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return keep
