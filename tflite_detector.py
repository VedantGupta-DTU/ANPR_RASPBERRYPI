"""
TFLite YOLO Detector  —  TinyML Edition
=========================================
Ultra-lightweight YOLOv8 inference using TensorFlow Lite Runtime.
~2 MB runtime vs ~50 MB for onnxruntime.  Optimised for ARM (XNNPACK).

Designed for Jetson Orin Nano / Raspberry Pi edge deployment.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple


class TFLiteDetector:
    """
    TinyML YOLOv8 detector using tflite-runtime.

    Automatically selects the fastest available delegate:
      1. GPU delegate  (Jetson / Mali)
      2. XNNPACK       (ARM NEON — default, very fast)
      3. CPU fallback
    """

    def __init__(self, model_path: str, imgsz: int = 320):
        self.imgsz = imgsz

        # Import tflite-runtime (tiny) or fall back to full tensorflow
        try:
            import tflite_runtime.interpreter as tflite
            print("[TFLite] Using tflite-runtime (lightweight)")
        except ImportError:
            import tensorflow.lite as tflite
            print("[TFLite] Using full TensorFlow Lite")

        print(f"[TFLite] Loading {model_path} …")

        # Try GPU delegate first, then XNNPACK, then plain CPU
        interpreter = None
        for delegate_name, delegate_factory in [
            ("GPU", lambda: tflite.load_delegate('libGpuDelegate.so')),
            ("XNNPACK", None),  # XNNPACK is default in tflite-runtime
        ]:
            try:
                if delegate_factory:
                    delegate = delegate_factory()
                    interpreter = tflite.Interpreter(
                        model_path=model_path,
                        experimental_delegates=[delegate],
                        num_threads=2,
                    )
                    print(f"[TFLite] Using {delegate_name} delegate")
                else:
                    interpreter = tflite.Interpreter(
                        model_path=model_path,
                        num_threads=2,
                    )
                    print(f"[TFLite] Using XNNPACK (ARM-optimised CPU)")
                break
            except Exception:
                continue

        if interpreter is None:
            interpreter = tflite.Interpreter(
                model_path=model_path, num_threads=2)
            print("[TFLite] Using CPU fallback")

        self.interpreter = interpreter
        self.interpreter.allocate_tensors()

        # Cache input / output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        inp = self.input_details[0]
        out = self.output_details[0]
        print(f"[TFLite] Input:  {inp['shape']}  dtype={inp['dtype']}")
        print(f"[TFLite] Output: {out['shape']}  dtype={out['dtype']}")

        # Detect input format: NHWC vs NCHW
        self._input_shape = tuple(inp['shape'])
        self._input_dtype = inp['dtype']
        # Check if input is NHWC (batch, H, W, 3) or NCHW (batch, 3, H, W)
        self._is_nhwc = (len(self._input_shape) == 4 and
                         self._input_shape[-1] == 3)

        # Detect if model needs uint8 or float32 input
        self._is_quantized = (inp['dtype'] == np.uint8 or
                              inp['dtype'] == np.int8)
        if self._is_quantized:
            self._input_scale = inp['quantization'][0]
            self._input_zero_point = inp['quantization'][1]
            print(f"[TFLite] Quantized model (INT8) — fastest on ARM!")
        else:
            print(f"[TFLite] Float32 model")

        print(f"[TFLite] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray,
               conf: float = 0.5,
               iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR numpy frame.

        Returns:
            List of dicts with 'bbox' [x1,y1,x2,y2] and 'confidence'.
        """
        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocess
        blob, ratio, (pad_w, pad_h) = self._preprocess(frame)

        # 2. Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], blob)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize output if needed
        if self.output_details[0]['dtype'] != np.float32:
            out_scale = self.output_details[0]['quantization'][0]
            out_zp = self.output_details[0]['quantization'][1]
            preds = (preds.astype(np.float32) - out_zp) * out_scale

        # 3. Postprocess
        return self._postprocess(
            preds, conf, iou_thresh, ratio, pad_w, pad_h, orig_w, orig_h)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize + normalize."""
        img_h, img_w = frame.shape[:2]
        target = self.imgsz

        ratio = min(target / img_w, target / img_h)
        new_w = int(round(img_w * ratio))
        new_h = int(round(img_h * ratio))

        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        pad_w = (target - new_w) // 2
        pad_h = (target - new_h) // 2
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR → RGB
        canvas = canvas[:, :, ::-1]

        if self._is_quantized:
            # INT8 quantized: scale input
            blob = ((canvas.astype(np.float32) / 255.0) /
                    self._input_scale + self._input_zero_point)
            blob = blob.astype(self._input_dtype)
        else:
            # Float32
            blob = canvas.astype(np.float32) / 255.0

        if self._is_nhwc:
            # TFLite expects NHWC: [1, H, W, 3]
            blob = np.expand_dims(blob, axis=0)
        else:
            # NCHW: [1, 3, H, W]
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

        return blob, ratio, (pad_w, pad_h)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(self, preds: np.ndarray,
                     conf_thresh: float, iou_thresh: float,
                     ratio: float, pad_w: int, pad_h: int,
                     orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        """
        Decode TFLite YOLO output. Handles:
          - End-to-end NMS format: [1, N, 6]  (x1,y1,x2,y2,score,class)
          - Raw format: [1, N, 4+nc] or [1, 4+nc, N]
        """
        preds = preds[0]  # squeeze batch

        # ── Format A: End-to-end NMS [N, 6] ──
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

        # ── Format B: Raw predictions ──
        # Could be [N, 4+nc] or [4+nc, N]
        if preds.shape[0] < preds.shape[1]:
            # [4+nc, N] → transpose to [N, 4+nc]
            preds = preds.T

        # preds is now [N, 4+nc]
        boxes_raw = preds[:, :4]   # cx, cy, w, h
        scores_raw = preds[:, 4:]  # class scores

        if scores_raw.shape[1] > 1:
            class_scores = np.max(scores_raw, axis=1)
        else:
            class_scores = scores_raw[:, 0]

        mask = class_scores >= conf_thresh
        if not np.any(mask):
            return []

        boxes_raw = boxes_raw[mask]
        class_scores = class_scores[mask]

        cx, cy, w, h = (boxes_raw[:, 0], boxes_raw[:, 1],
                        boxes_raw[:, 2], boxes_raw[:, 3])
        x1 = (cx - w / 2 - pad_w) / ratio
        y1 = (cy - h / 2 - pad_h) / ratio
        x2 = (cx + w / 2 - pad_w) / ratio
        y2 = (cy + h / 2 - pad_h) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        dets = np.stack([x1, y1, x2, y2, class_scores], axis=1)
        keep = self._nms_numpy(dets, iou_thresh)
        dets = dets[keep]

        results = []
        for det in dets:
            results.append({
                "bbox": [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
                "confidence": float(det[4]),
            })
        return results

    # ------------------------------------------------------------------
    # NMS (pure numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def _nms_numpy(dets: np.ndarray, iou_thresh: float) -> List[int]:
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
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
