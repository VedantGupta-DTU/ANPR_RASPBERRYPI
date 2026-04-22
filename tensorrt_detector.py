"""
TensorRT YOLO Detector  —  GPU-Accelerated Edition
====================================================
High-performance YOLOv8 inference using NVIDIA TensorRT.
Converts ONNX → TensorRT engine on first run, then uses GPU natively.

Designed for Jetson Orin Nano / Jetson AGX — requires JetPack 5+ or 6+.

Performance: ~15ms per frame (vs ~1000ms on CPU ONNX Runtime)
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple


class TensorRTDetector:
    """
    TensorRT YOLOv8 detector with automatic ONNX → Engine conversion.

    Priority order for model loading:
      1. Pre-built .engine file (instant load)
      2. Auto-convert from .onnx → .engine (one-time ~2 min build)

    Supports FP16 precision (default) for Tensor Core acceleration.
    """

    def __init__(self, model_path: str, imgsz: int = 640,
                 fp16: bool = True):
        self.imgsz = imgsz
        self.fp16 = fp16

        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Determine paths
        base, ext = os.path.splitext(model_path)
        if ext == '.engine':
            engine_path = model_path
            onnx_path = base + '.onnx'
        else:
            # .onnx passed
            onnx_path = model_path
            suffix = '_fp16' if fp16 else '_fp32'
            engine_path = base + suffix + '.engine'

        # Build engine if needed
        if os.path.exists(engine_path):
            print(f"[TRT] Loading cached engine: {engine_path}")
            self.engine = self._load_engine(engine_path)
        elif os.path.exists(onnx_path):
            print(f"[TRT] Building engine from {onnx_path} (one-time, ~2 min) ...")
            self.engine = self._build_engine(onnx_path, engine_path, fp16)
            print(f"[TRT] Engine saved to {engine_path}")
        else:
            raise FileNotFoundError(
                f"No .engine or .onnx found at {model_path}")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate GPU buffers
        self._setup_buffers()

        print(f"[TRT] Ready. (FP16={fp16}, imgsz={imgsz})")

    # ------------------------------------------------------------------
    # Engine build / load
    # ------------------------------------------------------------------

    def _build_engine(self, onnx_path: str, engine_path: str,
                      fp16: bool) -> 'trt.ICudaEngine':
        trt = self.trt
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT] Parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        # Use 1GB workspace (safe for 8GB Jetson)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     1 << 30)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled — using Tensor Cores")
        elif fp16:
            print("[TRT] WARNING: FP16 not supported, using FP32")

        # Build serialized engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        # Save engine for future loads
        with open(engine_path, 'wb') as f:
            f.write(serialized)

        # Deserialize
        runtime = trt.Runtime(self.logger)
        return runtime.deserialize_cuda_engine(serialized)

    def _load_engine(self, engine_path: str) -> 'trt.ICudaEngine':
        trt = self.trt
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            return runtime.deserialize_cuda_engine(f.read())

    # ------------------------------------------------------------------
    # Buffer setup (CUDA memory)
    # ------------------------------------------------------------------

    def _setup_buffers(self):
        """Allocate host and device memory for input/output tensors."""
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initializes CUDA context

        self.cuda = cuda
        self._bindings = []
        self._inputs = []
        self._outputs = []
        self._stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            # Map TRT dtype to numpy
            if dtype == self.trt.float16:
                np_dtype = np.float16
            elif dtype == self.trt.float32:
                np_dtype = np.float32
            elif dtype == self.trt.int32:
                np_dtype = np.int32
            else:
                np_dtype = np.float32

            size = abs(int(np.prod(shape)))
            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._bindings.append(int(device_mem))

            tensor_info = {
                'name': name,
                'shape': tuple(shape),
                'dtype': np_dtype,
                'host': host_mem,
                'device': device_mem,
            }

            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self._inputs.append(tensor_info)
                self.context.set_tensor_address(name, int(device_mem))
            else:
                self._outputs.append(tensor_info)
                self.context.set_tensor_address(name, int(device_mem))

        print(f"[TRT] Input:  {self._inputs[0]['shape']}")
        print(f"[TRT] Output: {self._outputs[0]['shape']}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray,
               conf: float = 0.5,
               iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR numpy frame.
        Returns list of dicts with 'bbox' [x1,y1,x2,y2] and 'confidence'.
        """
        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocess
        blob, ratio, (pad_w, pad_h) = self._preprocess(frame)

        # 2. Copy input to GPU
        inp = self._inputs[0]
        np.copyto(inp['host'], blob.ravel())
        self.cuda.memcpy_htod_async(inp['device'], inp['host'],
                                     self._stream)

        # 3. Run inference
        self.context.execute_async_v3(stream_handle=self._stream.handle)

        # 4. Copy output from GPU
        out = self._outputs[0]
        self.cuda.memcpy_dtoh_async(out['host'], out['device'],
                                     self._stream)
        self._stream.synchronize()

        # 5. Reshape and postprocess
        preds = out['host'].reshape(out['shape'])
        if preds.dtype == np.float16:
            preds = preds.astype(np.float32)

        return self._postprocess(
            preds, conf, iou_thresh, ratio, pad_w, pad_h, orig_w, orig_h)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize + normalize for NCHW input."""
        img_h, img_w = frame.shape[:2]
        target = self.imgsz

        ratio = min(target / img_w, target / img_h)
        new_w, new_h = int(round(img_w * ratio)), int(round(img_h * ratio))
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        pad_w = (target - new_w) // 2
        pad_h = (target - new_h) // 2
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR → RGB, HWC → CHW, normalize, add batch dim
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # CHW
        blob = np.expand_dims(blob, axis=0)    # NCHW
        blob = np.ascontiguousarray(blob)

        if self.fp16:
            blob = blob.astype(np.float16)

        return blob, ratio, (pad_w, pad_h)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(self, preds: np.ndarray,
                     conf_thresh: float, iou_thresh: float,
                     ratio: float, pad_w: int, pad_h: int,
                     orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        """
        Decode TensorRT YOLO output. Handles:
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
                x1 = max(0, min((float(x1_r) - pad_w) / ratio, orig_w))
                y1 = max(0, min((float(y1_r) - pad_h) / ratio, orig_h))
                x2 = max(0, min((float(x2_r) - pad_w) / ratio, orig_w))
                y2 = max(0, min((float(y2_r) - pad_h) / ratio, orig_h))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                results.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(score),
                })
            return results

        # ── Format B: Raw predictions ──
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # → [N, 4+nc]

        boxes_raw = preds[:, :4]
        scores_raw = preds[:, 4:]
        class_scores = (np.max(scores_raw, axis=1)
                        if scores_raw.shape[1] > 1 else scores_raw[:, 0])

        mask = class_scores >= conf_thresh
        if not np.any(mask):
            return []

        boxes_raw, class_scores = boxes_raw[mask], class_scores[mask]
        cx, cy, w, h = (boxes_raw[:, 0], boxes_raw[:, 1],
                        boxes_raw[:, 2], boxes_raw[:, 3])
        x1 = np.clip((cx - w / 2 - pad_w) / ratio, 0, orig_w)
        y1 = np.clip((cy - h / 2 - pad_h) / ratio, 0, orig_h)
        x2 = np.clip((cx + w / 2 - pad_w) / ratio, 0, orig_w)
        y2 = np.clip((cy + h / 2 - pad_h) / ratio, 0, orig_h)

        dets = np.stack([x1, y1, x2, y2, class_scores], axis=1)
        keep = self._nms(dets, iou_thresh)
        return [{"bbox": [int(d[0]), int(d[1]), int(d[2]), int(d[3])],
                 "confidence": float(d[4])} for d in dets[keep]]

    @staticmethod
    def _nms(dets: np.ndarray, thresh: float) -> List[int]:
        x1, y1, x2, y2, scores = (dets[:, 0], dets[:, 1],
                                    dets[:, 2], dets[:, 3], dets[:, 4])
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
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= thresh)[0] + 1]
        return keep
