"""
Industry-Grade Video ANPR Pipeline
===================================
Processes CCTV / dashcam video for license plate recognition using
industry best practices:

  1. In-memory frame pipeline   – no disk I/O per frame
  2. Direct numpy inference      – YOLO + OCR accept numpy arrays
  3. Temporal voting              – groups OCR reads across frames by
                                    IoU overlap, picks consensus plate
  4. Per-stage latency profiling  – detection, crop, OCR, total (with
                                    min / avg / max / P95 stats)

Usage:
    python video_pipeline.py \
        --input "test_videos/myfile.mp4" \
        --engine paddle \
        --frame-skip 3
"""

import os
import sys
import csv
import json
import time
import argparse
import statistics
import datetime
import threading as _threading
from typing import List, Dict, Any, Optional, Tuple

import requests as _requests

import cv2
import numpy as np
import re

import config
from indian_plate_formatter import IndianPlateFormatter

# Database module (optional — graceful degradation if not available)
try:
    import db as anpr_db
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_iou(box1: List[int], box2: List[int]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def _bbox_center(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_diagonal(bbox: List[int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def _enhance_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE enhancement for low-light frames."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def _preprocess_crop(crop: np.ndarray, fast: bool = False) -> List[np.ndarray]:
    """
    Quick in-memory preprocessing variants for a plate crop.
    In fast mode: returns only [original, CLAHE] (2 variants).
    In full mode: returns [original, CLAHE, sharpened, adaptive, otsu, denoised] (6+ variants).
    """
    variants: List[np.ndarray] = [crop]
    h, w = crop.shape[:2]

    # Up-scale small plates more aggressively for better OCR
    if w < 300 or h < 80:
        scale = max(300 / max(w, 1), 80 / max(h, 1), 2.0)
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        variants[0] = crop  # replace original with upscaled

    # CLAHE
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    # Sharpen (helps distinguish 8/5, R/H, 0/D etc.)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    variants.append(cv2.filter2D(crop, -1, kernel))

    # In fast mode, stop here (3 variants: original, CLAHE, sharpened)
    if fast:
        return variants

    # ---- Full accuracy mode: additional variants ----

    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    variants.append(cv2.filter2D(crop, -1, kernel))

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

    # Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    # Denoised grayscale
    denoised = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    variants.append(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))

    # Bike plates are often 2-line; split and vote easier.
    aspect_ratio = h / w if w > 0 else 0
    if aspect_ratio > 0.7:
        mid_y = h // 2
        overlap = max(int(h * 0.25), 8)
        top_half = crop[0 : mid_y + overlap, :]
        bottom_half = crop[mid_y - overlap : h, :]

        # Top/bottom originals + quick CLAHE for each
        variants.append(top_half)
        variants.append(bottom_half)

        top_gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
        bot_gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        variants.append(cv2.cvtColor(clahe.apply(top_gray), cv2.COLOR_GRAY2BGR))
        variants.append(cv2.cvtColor(clahe.apply(bot_gray), cv2.COLOR_GRAY2BGR))

    return variants


# ---------------------------------------------------------------------------
# OCR wrapper – works entirely in memory (no disk writes)
# ---------------------------------------------------------------------------

class InMemoryOCR:
    """
    Wraps PaddleOCR, RapidOCR, or EasyOCR for in-memory numpy-array input.
    All heavy OCR readers in ocr_reader.py ultimately call methods that
    accept numpy arrays, so we wire that up directly here to avoid the
    disk round-trip that the existing pipeline uses.

    Engine options:
      - 'rapidocr' : PaddleOCR models via ONNX Runtime (works on ARM64/Pi!)
      - 'paddle'   : Native PaddlePaddle (x86 only, segfaults on ARM64)
      - 'easyocr'  : EasyOCR (lighter but less accurate)
    """

    def __init__(self, engine: str = "rapidocr"):
        self.engine_name = engine
        self._ocr = None
        self._formatter = IndianPlateFormatter()

    def load(self):
        if self._ocr is not None:
            return
        if self.engine_name == "paddle":
            import os as _os
            _os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
            from paddleocr import PaddleOCR
            print("[OCR] Loading PaddleOCR …")
            self._ocr = PaddleOCR(lang='en')
            print("[OCR] PaddleOCR ready.")
        elif self.engine_name == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR
            print("[OCR] Loading RapidOCR (PaddleOCR models via ONNX Runtime) …")
            self._ocr = RapidOCR()
            print("[OCR] RapidOCR ready.")
        elif self.engine_name == "easyocr":
            import easyocr
            import torch
            print("[OCR] Loading EasyOCR …")
            self._ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("[OCR] EasyOCR ready.")
        else:
            raise ValueError(f"Unsupported engine: {self.engine_name}")

    # ---- core read ----------------------------------------------------------

    def read_numpy(self, crop_bgr: np.ndarray) -> Tuple[str, bool, float]:
        """
        Run OCR on a numpy crop (BGR). Returns (plate_text, is_valid).
        Tries multiple preprocessing variants and picks the best valid read.
        """
        if self._ocr is None:
            self.load()

        fast = getattr(config, 'FAST_MODE', False)
        variants = _preprocess_crop(crop_bgr, fast=fast)

        best_text = ""
        best_score = -1.0
        best_valid = False
        best_ocr_conf = 0.0

        # Common OCR confusions we want to treat as “close enough” when
        # choosing between multiple valid formatted plates.
        conf_pairs = {
            ("O", "D"), ("D", "O"),
            ("0", "D"), ("D", "0"),
            ("0", "O"), ("O", "0"),
            ("N", "W"), ("W", "N"),
            ("6", "8"), ("8", "6"),
        }
        conf_neighbors: Dict[str, List[str]] = {}
        for a, b in conf_pairs:
            conf_neighbors.setdefault(a, []).append(b)

        # Directional weights: if raw OCR seems to say X but candidate is Y,
        # we consider X->Y more likely (based on your observed errors).
        # This nudges selection toward expected values for common swaps.
        # Set weights > 1.0 if we want an auto-correction to completely OVERRIDE the raw OCR match.
        directional_weights: Dict[Tuple[str, str], float] = {
            ("O", "D"): 1.10,  # Strongly favor D over raw O
            ("N", "W"): 1.10,  # Strongly favor W over raw N
            # For raw '0', we prefer 'D' over 'O', but neither beats an exact '0' if '0' is valid
            ("0", "D"): 0.95,  
            ("0", "O"): 0.85,
            # Keep 8->6 slightly favored but not overriding an exact 8 match
            ("8", "6"): 0.85,
            # reverse direction is treated as weaker
            ("D", "O"): 0.70,
            ("D", "0"): 0.70,
            ("W", "N"): 0.70,
            ("6", "8"): 0.70,
        }

        def char_similarity(a: str, b: str) -> Tuple[float, int, int, int]:
            """
            Similarity in [0,1] between cleaned raw OCR and formatted candidate.
            Uses a small confusion map (e.g. O<->D, N<->W) to avoid over-penalizing
            common misreads.
            """
            if not a or not b:
                return 0.0, 0, 0, 0
            max_len = max(len(a), len(b))
            a = a.ljust(max_len, "_")
            b = b.ljust(max_len, "_")

            sim = 0.0
            nw_matches = 0
            od_matches = 0
            eight_six_matches = 0
            for ca, cb in zip(a, b):
                if ca == cb:
                    sim += 1.0
                elif (ca, cb) in directional_weights:
                    sim += directional_weights[(ca, cb)]
                    if (ca, cb) == ("N", "W"):
                        nw_matches += 1
                    elif (ca, cb) == ("O", "D"):
                        od_matches += 1
                    elif (ca, cb) == ("8", "6"):
                        eight_six_matches += 1
                elif (ca, cb) in conf_pairs:
                    sim += 0.7
                else:
                    sim += 0.0
            return (sim / max_len if max_len else 0.0), nw_matches, od_matches, eight_six_matches

        for var in variants:
            raw, ocr_conf = self._run_engine(var)
            if not raw:
                continue
            is_valid, formatted = self._formatter.validate_plate(raw)
            raw_clean = re.sub(r"[^A-Z0-9]", "", raw.upper())

            # Evaluate candidate formatted plates:
            # - Always evaluate the original formatted output.
            # - If the original formatted is valid, also evaluate a small number of
            #   alternatives created by common confusion swaps (N<->W, 8<->6, O<->D).
            candidate_formatted: List[Tuple[str, bool]] = [(formatted, is_valid)]

            if is_valid:
                fmt_clean = re.sub(r"[^A-Z0-9]", "", formatted.upper().replace(" ", ""))
                candidate_clean_set = {fmt_clean}

                # Single swaps (any position)
                for i, ch in enumerate(fmt_clean):
                    if ch in conf_neighbors:
                        for repl in conf_neighbors[ch]:
                            cand = fmt_clean[:i] + repl + fmt_clean[i + 1 :]
                            candidate_clean_set.add(cand)

                # Two-swap combos: one “letter” confusion + one “digit” confusion
                # This targets things like CAN9288 -> CAW9268.
                letter_positions = [i for i, ch in enumerate(fmt_clean) if ch in ("N", "W", "O", "D")]
                digit_positions = [i for i, ch in enumerate(fmt_clean) if ch in ("6", "8")]

                for li in letter_positions:
                    lch = fmt_clean[li]
                    for lrepl in conf_neighbors.get(lch, []):
                        tmp = fmt_clean[:li] + lrepl + fmt_clean[li + 1 :]
                        for di in digit_positions:
                            dch = tmp[di]
                            for drepl in conf_neighbors.get(dch, []):
                                cand = tmp[:di] + drepl + tmp[di + 1 :]
                                candidate_clean_set.add(cand)

                # Validate each alternative and add to candidate list
                for cand_clean in candidate_clean_set:
                    alt_valid, alt_formatted = self._formatter.validate_plate(cand_clean)
                    if alt_valid:
                        candidate_formatted.append((alt_formatted, True))

            # Choose best among candidates
            for cand_formatted, cand_valid in candidate_formatted:
                cand_clean = re.sub(r"[^A-Z0-9]", "", cand_formatted.upper())
                sim, nw_matches, od_matches, eight_six_matches = char_similarity(raw_clean, cand_clean)
                # Multiplicative similarity term + directional bonus.
                # Directional bonus nudges toward the common corrected swaps
                # (N->W, 8->6, O->D) when both original and alternative are valid.
                denom = max(1, len(cand_clean))
                directional_bonus = (nw_matches + od_matches) / denom
                # Only strongly apply 8->6 bonus if we are also doing N->W correction.
                if nw_matches > 0:
                    directional_bonus += 2.0 * eight_six_matches / denom
                effective_ocr_conf = float(ocr_conf) * (0.5 + sim) + float(ocr_conf) * 0.15 * directional_bonus

                score = (
                    (1000.0 * effective_ocr_conf + len(cand_clean)) if cand_valid
                    else (100.0 * effective_ocr_conf + len(cand_clean))
                )

                if score > best_score:
                    best_score = score
                    best_text = cand_formatted
                    best_valid = cand_valid
                    best_ocr_conf = effective_ocr_conf

        return (best_text or "UNREADABLE", best_valid, best_ocr_conf)

    # ---- engine dispatch ----------------------------------------------------

    def _run_engine(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        if self.engine_name == "paddle":
            return self._paddle_read(img_bgr)
        elif self.engine_name == "rapidocr":
            return self._rapidocr_read(img_bgr)
        else:
            return self._easyocr_read(img_bgr)

    def _paddle_read(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        result = self._ocr.predict(img_bgr)
        for res in result:
            if 'rec_texts' in res:
                texts = []
                scores = res.get('rec_scores', [1.0] * len(res['rec_texts']))
                kept_scores: List[float] = []
                for txt, sc in zip(res['rec_texts'], scores):
                    t = txt.upper().strip()
                    if t in ('IND', 'INDIA', 'IN', 'ONI', 'ND', 'INO'):
                        continue
                    if sc < 0.3:
                        continue
                    texts.append(t)
                    kept_scores.append(float(sc))
                if not texts:
                    return ("", 0.0)
                # Confidence proxy: average recognition score of kept tokens.
                ocr_conf = float(sum(kept_scores) / max(1, len(kept_scores)))
                return (' '.join(texts), ocr_conf)
        return ("", 0.0)

    def _easyocr_read(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        results = self._ocr.readtext(img_bgr)
        texts: List[str] = []
        scores: List[float] = []
        for r in results:
            txt = (r[1] or "").upper().strip()
            if not txt:
                continue
            texts.append(txt)
            try:
                scores.append(float(r[2]))
            except Exception:
                pass
        if not texts:
            return ("", 0.0)
        ocr_conf = float(sum(scores) / max(1, len(scores))) if scores else 0.0
        return (' '.join(texts), ocr_conf)

    def _rapidocr_read(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        """RapidOCR: same PaddleOCR models, ONNX Runtime backend."""
        result, _ = self._ocr(img_bgr)
        if not result:
            return ("", 0.0)
        texts: List[str] = []
        scores: List[float] = []
        for line in result:
            # Each line: [bbox, text, confidence]
            txt = (line[1] or "").upper().strip()
            if not txt:
                continue
            if txt in ('IND', 'INDIA', 'IN', 'ONI', 'ND', 'INO'):
                continue
            conf = float(line[2]) if len(line) > 2 else 0.0
            if conf < 0.3:
                continue
            texts.append(txt)
            scores.append(conf)
        if not texts:
            return ("", 0.0)
        ocr_conf = float(sum(scores) / max(1, len(scores)))
        return (' '.join(texts), ocr_conf)


# ---------------------------------------------------------------------------
# Plate Track – temporal voting across frames
# ---------------------------------------------------------------------------

class PlateTrack:
    """Accumulates OCR reads for the same physical plate across frames."""

    def __init__(
        self,
        bbox: List[int],
        text: str,
        is_valid: bool,
        det_conf: float,
        ocr_conf: float,
        frame_idx: int,
        time_sec: float,
    ):
        self.start_frame_idx = frame_idx
        self.bboxes: List[List[int]] = [bbox]
        self.reads: List[Dict[str, Any]] = [{
            "text": text, "is_valid": is_valid,
            "det_conf": det_conf, "ocr_conf": ocr_conf, "frame_idx": frame_idx,
            "time_sec": time_sec,
        }]
        self.last_frame_idx = frame_idx

    @property
    def last_bbox(self) -> List[int]:
        return self.bboxes[-1]

    def add(self, bbox, text, is_valid, det_conf, ocr_conf, frame_idx, time_sec):
        self.bboxes.append(bbox)
        self.last_frame_idx = frame_idx
        self.reads.append({
            "text": text, "is_valid": is_valid,
            "det_conf": det_conf, "ocr_conf": ocr_conf, "frame_idx": frame_idx,
            "time_sec": time_sec,
        })

    def best_read(self) -> Dict[str, Any]:
        """Pick the best OCR read via temporal voting."""
        # Use all reads in the track! Weighted scoring handles blurry frames naturally.
        # Use all reads in the track! Weighted scoring handles blurry frames naturally.
        reads_sorted = sorted(self.reads, key=lambda r: r["frame_idx"])
        if not reads_sorted:
            return {"text": "UNREADABLE", "is_valid": False, "det_conf": 0.0}

        tail_reads = reads_sorted

        # Prefer valid reads within the track, but fall back to all reads.
        valid = [r for r in tail_reads if r["is_valid"]]
        pool = valid if valid else tail_reads

        # Weighted vote: prefer detections with strong YOLO confidence and strong OCR confidence.
        det_conf_power = 1.0
        ocr_conf_power = float(getattr(config, "VIDEO_OCR_CONF_POWER", 1.0))
        weight_sum: Dict[str, float] = {}
        freq: Dict[str, int] = {}

        for r in pool:
            key = (r.get("text") or "").replace(" ", "")
            if not key:
                continue
            det_c = float(r.get("det_conf", 0.0))
            ocr_c = float(r.get("ocr_conf", 0.0))
            w = (det_c ** det_conf_power) * (ocr_c ** ocr_conf_power)
            weight_sum[key] = weight_sum.get(key, 0.0) + w
            freq[key] = freq.get(key, 0) + 1

        if not weight_sum:
            # Fallback: pick most frequent key (should be rare).
            freq = {k: v for k, v in freq.items()}
            best_key = max(freq, key=freq.get) if freq else ""
        else:
            # Primary selection: highest det_conf-weighted support.
            best_key = max(weight_sum, key=weight_sum.get)

        # Among reads matching that key, pick highest detection confidence,
        # then OCR confidence, then later frame.
        candidates = [r for r in pool if (r.get("text") or "").replace(" ", "") == best_key]
        best = max(
            candidates,
            key=lambda r: (
                float(r.get("det_conf", 0.0)),
                float(r.get("ocr_conf", 0.0)),
                int(r.get("frame_idx", 0)),
            ),
        )

        best_key_freq = freq.get(best_key, 0)
        stability_ratio = (best_key_freq / len(pool)) if pool else 0.0
        best_key_weight = weight_sum.get(best_key, 0.0)

        # Add stats for downstream confirmation.
        best = {**best,
                "best_key": best_key,
                "best_key_freq": best_key_freq,
                "best_key_ratio": stability_ratio,
                "best_key_weight": best_key_weight,
                "pool_len": len(pool),
                "valid_reads_len": len(valid)}
        return best


# ---------------------------------------------------------------------------
# Main Video Pipeline
# ---------------------------------------------------------------------------

class VideoPipeline:
    """Industry-grade video ANPR with per-stage latency profiling."""

    def __init__(self, engine: str = "paddle", frame_skip: int = None):
        self.frame_skip = frame_skip or getattr(config, "VIDEO_FRAME_SKIP", 2)
        self.fast_mode = getattr(config, 'FAST_MODE', False)
        self.yolo_imgsz = getattr(config, 'YOLO_IMGSZ', 640)

        # Detection — priority: TensorRT GPU > ONNX+CUDA > TFLite > ONNX CPU
        engine_path = getattr(config, 'MODEL_PATH_ENGINE', '')
        tflite_path = getattr(config, 'MODEL_PATH_TFLITE', '')
        onnx_path = getattr(config, 'MODEL_PATH_ONNX', '')
        if not onnx_path or not os.path.exists(onnx_path):
            onnx_path = config.MODEL_PATH.replace('.pt', '.onnx')

        self.yolo = None
        detector_name = "none"

        # 1. TensorRT (Jetson GPU — fastest)
        if self.yolo is None:
            try:
                # Check if engine exists OR if we can build from ONNX
                can_build = os.path.exists(onnx_path)
                if (engine_path and os.path.exists(engine_path)) or can_build:
                    from tensorrt_detector import TensorRTDetector
                    src = engine_path if os.path.exists(engine_path) else onnx_path
                    self.yolo = TensorRTDetector(src, imgsz=self.yolo_imgsz)
                    detector_name = "TensorRT GPU"
            except Exception as e:
                print(f"[DET] TensorRT unavailable: {e}")

        # 2. ONNX Runtime + CUDA (Jetson GPU — good)
        if self.yolo is None and os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    from onnx_detector import ONNXDetector
                    self.yolo = ONNXDetector(onnx_path, imgsz=self.yolo_imgsz)
                    detector_name = "ONNX Runtime + CUDA"
            except Exception as e:
                print(f"[DET] ONNX CUDA unavailable: {e}")

        # 3. TFLite (ARM XNNPACK — lightweight)
        if self.yolo is None and tflite_path and os.path.exists(tflite_path):
            try:
                from tflite_detector import TFLiteDetector
                self.yolo = TFLiteDetector(tflite_path, imgsz=self.yolo_imgsz)
                detector_name = "TFLite XNNPACK"
            except Exception as e:
                print(f"[DET] TFLite unavailable: {e}")

        # 4. ONNX Runtime CPU (fallback)
        if self.yolo is None and os.path.exists(onnx_path):
            from onnx_detector import ONNXDetector
            self.yolo = ONNXDetector(onnx_path, imgsz=self.yolo_imgsz)
            detector_name = "ONNX Runtime CPU"

        if self.yolo is None:
            raise FileNotFoundError(
                f"No model found! Place best.engine, best.onnx, or best.tflite in {config.BASE_DIR}"
            )
        print(f"[DET] YOLO ready via {detector_name}. (imgsz={self.yolo_imgsz}, fast={self.fast_mode})")

        # OCR
        self.ocr = InMemoryOCR(engine=engine)
        self.ocr.load()

        self._formatter = IndianPlateFormatter()

        # Latency accumulators
        self.timings: Dict[str, List[float]] = {
            "detect": [], "crop": [], "ocr": [], "total": [],
        }

    # ------------------------------------------------------------------

    def process(self, video_path: str,
                csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_path)

        print(f"\n{'='*60}")
        print(f"  Video ANPR Pipeline  (industry-grade)")
        print(f"{'='*60}")
        print(f"  File        : {video_name}")
        print(f"  FPS         : {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frame skip  : {self.frame_skip}")
        print(f"  OCR engine  : {self.ocr.engine_name}")
        print(f"{'='*60}\n")

        from concurrent.futures import ThreadPoolExecutor, Future

        tracks: List[PlateTrack] = []
        frame_idx = 0
        processed_count = 0

        # Tracking association settings:
        track_iou_assign_min = 0.20
        track_max_age_frames = 15
        dist_penalty_weight = 0.15

        # Async OCR: pending futures from previous frames
        # Each item: (future, det_bbox, det_conf, frame_idx, timestamp_sec)
        pending_ocr: List[Tuple[Future, List[int], float, int, float]] = []

        ocr_pool = ThreadPoolExecutor(max_workers=2)

        def _associate_to_track(det_bbox, text, is_valid, det_conf, ocr_conf,
                                f_idx, t_sec):
            """Associate a detection+OCR result with an existing track or create new."""
            best_track = None
            best_score = float("-inf")
            det_center = _bbox_center(det_bbox)
            det_diag = _bbox_diagonal(det_bbox)

            for track in tracks:
                if hasattr(track, "last_frame_idx"):
                    if f_idx - track.last_frame_idx > track_max_age_frames:
                        continue
                iou = _calculate_iou(det_bbox, track.last_bbox)
                if iou < track_iou_assign_min:
                    continue
                track_center = _bbox_center(track.last_bbox)
                dist = float(np.sqrt((det_center[0] - track_center[0]) ** 2 +
                                      (det_center[1] - track_center[1]) ** 2))
                dist_norm = dist / (det_diag + 1e-6)
                score = iou - (dist_penalty_weight * dist_norm)
                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track is not None:
                best_track.add(det_bbox, text, is_valid, det_conf, ocr_conf,
                               f_idx, t_sec)
            else:
                tracks.append(PlateTrack(
                    det_bbox, text, is_valid, det_conf, ocr_conf, f_idx, t_sec))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue

                t_frame_start = time.perf_counter()
                timestamp_sec = frame_idx / fps

                # --- Collect completed OCR futures from previous frames ---
                still_pending = []
                for fut, bbox, conf, f_idx, t_sec in pending_ocr:
                    if fut.done():
                        text, is_valid, ocr_conf = fut.result()
                        _associate_to_track(bbox, text, is_valid, conf,
                                            ocr_conf, f_idx, t_sec)
                    else:
                        still_pending.append((fut, bbox, conf, f_idx, t_sec))
                pending_ocr = still_pending

                # --- Stage 1: Detection ---
                t0 = time.perf_counter()
                detections = self._detect_in_memory(frame)
                t_detect = time.perf_counter() - t0

                # --- Stage 2 & 3: Crop + submit OCR async ---
                t_crop_total = 0.0

                for det in detections:
                    tc0 = time.perf_counter()
                    crop = self._crop_plate(frame, det["bbox"])
                    t_crop_total += time.perf_counter() - tc0

                    if getattr(config, "IGNORE_BIKE_PLATES", False):
                        crop_h, crop_w = crop.shape[:2]
                        if crop_w > 0 and (crop_h / crop_w) > 0.65:
                            continue

                    # Queue limiter: drop new crops if too many OCR tasks pending
                    # (prevents CPU/memory exhaustion on Pi / edge devices)
                    MAX_PENDING_OCR = getattr(config, "MAX_PENDING_OCR", 5)
                    if len(pending_ocr) >= MAX_PENDING_OCR:
                        continue  # skip this crop — tracker will retry next frame

                    # Submit OCR to thread pool (non-blocking)
                    future = ocr_pool.submit(self.ocr.read_numpy, crop)
                    pending_ocr.append((
                        future, det["bbox"], det["confidence"],
                        frame_idx, timestamp_sec
                    ))

                t_total = time.perf_counter() - t_frame_start

                # Record timings (OCR is async, so we only record detect+crop)
                self.timings["detect"].append(t_detect)
                self.timings["crop"].append(t_crop_total)
                self.timings["ocr"].append(0.0)  # OCR runs in background
                self.timings["total"].append(t_total)

                processed_count += 1
                if processed_count % 25 == 0:
                    print(f"  [frame {frame_idx:5d}]  detections={len(detections)}  "
                          f"total={t_total*1000:.1f}ms  tracks={len(tracks)}  "
                          f"pending_ocr={len(pending_ocr)}")

                frame_idx += 1

        finally:
            # Drain all remaining OCR futures
            for fut, bbox, conf, f_idx, t_sec in pending_ocr:
                try:
                    text, is_valid, ocr_conf = fut.result(timeout=10)
                    _associate_to_track(bbox, text, is_valid, conf,
                                        ocr_conf, f_idx, t_sec)
                except Exception:
                    pass
            ocr_pool.shutdown(wait=True)

        cap.release()

        # --- Track post-processing: merge fragments → confirmed plates ---
        tracks = self._merge_track_fragments(tracks)
        confirmed = self._resolve_tracks(tracks, video_name)

        # --- Output ---
        if csv_path is None:
            base = os.path.splitext(video_name)[0]
            csv_path = os.path.join(config.OUTPUT_DIR, f"{base}_plates.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._write_csv(confirmed, csv_path)

        # Latency report
        self._print_latency_report(processed_count)

        return confirmed

    # ------------------------------------------------------------------
    # Live camera pipeline
    # ------------------------------------------------------------------

    def _send_to_api(self, plate_text: str, confidence: float,
                     camera_id: str, api_endpoint: str):
        """POST a confirmed plate to the external API (non-blocking)."""
        vehicle_no = plate_text.replace(" ", "")
        payload = {
            "camera_id": camera_id,
            "vehicle_no": vehicle_no,
            "raw_plate": plate_text,
            "confidence": round(confidence, 4),
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        }

        def _post():
            try:
                resp = _requests.post(
                    api_endpoint, json=payload, timeout=5)
                print(f"  [API] → {api_endpoint}  "
                      f"status={resp.status_code}  "
                      f"plate={vehicle_no}")
            except Exception as e:
                print(f"  [API] ✗ Failed to send {vehicle_no}: {e}")

        _threading.Thread(target=_post, daemon=True).start()

    def process_live(self, source=0,
                     csv_path: Optional[str] = None,
                     show_window: bool = True,
                     camera_id: str = "CAM_001",
                     api_endpoint: Optional[str] = None,
                     ) -> List[Dict[str, Any]]:
        """
        Live ANPR pipeline for real-time camera feeds.

        Args:
            source: Camera source — integer for webcam index (0, 1, ...),
                    or string for RTSP/HTTP URL.
            csv_path: Optional CSV path for logging confirmed plates.
            show_window: If True, display a live window with bounding boxes.
            camera_id: Camera identifier for the API payload.
            api_endpoint: URL to POST confirmed plates to. If None, no API call.

        Returns:
            List of all confirmed plates detected during the session.

        Controls:
            Press 'q' in the live window to stop.
        """
        # Parse source: try integer first (webcam), else treat as URL string
        try:
            source_val = int(source)
        except (ValueError, TypeError):
            source_val = str(source)

        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default for webcams that don't report FPS

        source_label = f"Camera {source_val}" if isinstance(source_val, int) \
            else source_val

        print(f"\n{'='*60}")
        print(f"  Live ANPR Pipeline  (real-time)")
        print(f"{'='*60}")
        print(f"  Source      : {source_label}")
        print(f"  FPS         : {fps:.1f}")
        print(f"  Frame skip  : {self.frame_skip}")
        print(f"  OCR engine  : {self.ocr.engine_name}")
        print(f"  Press 'q' to stop.")
        print(f"{'='*60}\n")

        # CSV setup — append mode for continuous logging
        if csv_path is None:
            csv_path = os.path.join(config.OUTPUT_DIR, "live_plates.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file,
            fieldnames=["plate", "det_conf", "ocr_conf", "num_reads",
                        "time_sec", "source"])
        csv_writer.writeheader()

        tracks: List[PlateTrack] = []
        all_confirmed: List[Dict[str, Any]] = []
        frame_idx = 0
        processed_count = 0

        # Tracking settings (same as video mode)
        track_iou_assign_min = 0.20
        track_max_age_frames = 15
        dist_penalty_weight = 0.15
        # For live mode, emit a plate when its track hasn't been updated
        # for this many processed frames:
        track_emit_age = 20

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    # Still show the frame (no detections drawn)
                    if show_window:
                        cv2.imshow("ANPR Live", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue

                t_frame_start = time.perf_counter()
                timestamp_sec = round(frame_idx / fps, 2)

                # --- Stage 1: Detection ---
                t0 = time.perf_counter()
                detections = self._detect_in_memory(frame)
                t_detect = time.perf_counter() - t0

                # --- Stage 2 & 3: Crop + OCR ---
                t_crop_total = 0.0
                t_ocr_total = 0.0
                frame_results = []  # For drawing

                for det in detections:
                    tc0 = time.perf_counter()
                    crop = self._crop_plate(frame, det["bbox"])
                    t_crop_total += time.perf_counter() - tc0

                    # Filter bike plates
                    if getattr(config, "IGNORE_BIKE_PLATES", False):
                        crop_h, crop_w = crop.shape[:2]
                        if crop_w > 0 and (crop_h / crop_w) > 0.65:
                            continue

                    to0 = time.perf_counter()
                    text, is_valid, ocr_conf = self.ocr.read_numpy(crop)
                    t_ocr_total += time.perf_counter() - to0

                    # Track association
                    best_track = None
                    best_score = float("-inf")
                    det_bbox = det["bbox"]
                    det_center = _bbox_center(det_bbox)
                    det_diag = _bbox_diagonal(det_bbox)

                    for track in tracks:
                        if hasattr(track, "last_frame_idx"):
                            if frame_idx - track.last_frame_idx > track_max_age_frames:
                                continue
                        iou = _calculate_iou(det_bbox, track.last_bbox)
                        if iou < track_iou_assign_min:
                            continue
                        track_center = _bbox_center(track.last_bbox)
                        dist = float(np.sqrt(
                            (det_center[0] - track_center[0]) ** 2 +
                            (det_center[1] - track_center[1]) ** 2))
                        dist_norm = dist / (det_diag + 1e-6)
                        score = iou - (dist_penalty_weight * dist_norm)
                        if score > best_score:
                            best_score = score
                            best_track = track

                    if best_track is not None:
                        best_track.add(det_bbox, text, is_valid,
                                       det["confidence"], ocr_conf,
                                       frame_idx, timestamp_sec)
                    else:
                        tracks.append(PlateTrack(
                            det_bbox, text, is_valid,
                            det["confidence"], ocr_conf,
                            frame_idx, timestamp_sec))

                    # Collect for drawing
                    frame_results.append({
                        "bbox": det_bbox,
                        "text": text if is_valid else "",
                        "conf": det["confidence"],
                    })

                t_total = time.perf_counter() - t_frame_start
                self.timings["detect"].append(t_detect)
                self.timings["crop"].append(t_crop_total)
                self.timings["ocr"].append(t_ocr_total)
                self.timings["total"].append(t_total)
                processed_count += 1

                # --- Emit confirmed plates from stale tracks ---
                still_active = []
                for track in tracks:
                    age = frame_idx - track.last_frame_idx
                    if age > track_emit_age:
                        # Track is stale — resolve it
                        best = track.best_read()
                        if (len(track.reads) >= getattr(config, "VIDEO_MIN_TRACK_READS", 2)
                                and best.get("is_valid")
                                and best.get("ocr_conf", 0) >= getattr(config, "VIDEO_MIN_OCR_CONF", 0)):
                            plate_text = best["text"]
                            # Deduplicate against already confirmed
                            if not any(c["plate"] == plate_text for c in all_confirmed):
                                entry = {
                                    "plate": plate_text,
                                    "det_conf": round(best.get("det_conf", 0), 3),
                                    "ocr_conf": round(best.get("ocr_conf", 0), 3),
                                    "num_reads": len(track.reads),
                                    "time_sec": best.get("time_sec", 0),
                                    "source": str(source_label),
                                }
                                all_confirmed.append(entry)
                                csv_writer.writerow(entry)
                                csv_file.flush()
                                print(f"  ✓ PLATE DETECTED: {plate_text}  "
                                      f"(conf={entry['det_conf']}, "
                                      f"reads={entry['num_reads']}, "
                                      f"t={entry['time_sec']}s)")
                                # Save to database
                                if _DB_AVAILABLE and getattr(config, 'DB_ENABLED', True):
                                    try:
                                        anpr_db.insert_detection(
                                            plate=plate_text,
                                            det_conf=entry['det_conf'],
                                            ocr_conf=entry['ocr_conf'],
                                            source=entry.get('source', str(source_label)),
                                            source_type='live',
                                            num_reads=entry['num_reads'],
                                            time_sec=entry['time_sec'],
                                            is_valid=True,
                                        )
                                    except Exception as e:
                                        print(f"  ⚠ DB write failed: {e}")
                                # POST to external API
                                if api_endpoint:
                                    self._send_to_api(
                                        plate_text, entry['det_conf'],
                                        camera_id, api_endpoint)
                    else:
                        still_active.append(track)
                tracks = still_active

                # --- Draw bounding boxes on frame ---
                if show_window:
                    display = frame.copy()
                    for r in frame_results:
                        x1, y1, x2, y2 = r["bbox"]
                        color = (0, 255, 0) if r["text"] else (0, 0, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        if r["text"]:
                            label = f"{r['text']} ({r['conf']:.2f})"
                            (tw, th), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display,
                                          (x1, y1 - th - 10),
                                          (x1 + tw, y1), color, -1)
                            cv2.putText(display, label,
                                        (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 0), 2)

                    # FPS overlay
                    live_fps = 1.0 / t_total if t_total > 0 else 0
                    cv2.putText(display,
                                f"FPS: {live_fps:.1f} | Plates: {len(all_confirmed)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2)

                    cv2.imshow("ANPR Live", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")

        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()

            # Resolve any remaining active tracks
            for track in tracks:
                best = track.best_read()
                if (len(track.reads) >= getattr(config, "VIDEO_MIN_TRACK_READS", 2)
                        and best.get("is_valid")
                        and best.get("ocr_conf", 0) >= getattr(config, "VIDEO_MIN_OCR_CONF", 0)):
                    plate_text = best["text"]
                    if not any(c["plate"] == plate_text for c in all_confirmed):
                        entry = {
                            "plate": plate_text,
                            "det_conf": round(best.get("det_conf", 0), 3),
                            "ocr_conf": round(best.get("ocr_conf", 0), 3),
                            "num_reads": len(track.reads),
                            "time_sec": best.get("time_sec", 0),
                            "source": str(source_label),
                        }
                        all_confirmed.append(entry)
                        csv_writer.writerow(entry)
                        print(f"  ✓ PLATE DETECTED: {plate_text}  "
                              f"(conf={entry['det_conf']}, "
                              f"reads={entry['num_reads']}, "
                              f"t={entry['time_sec']}s)")
                        # Save to database
                        if _DB_AVAILABLE and getattr(config, 'DB_ENABLED', True):
                            try:
                                anpr_db.insert_detection(
                                    plate=plate_text,
                                    det_conf=entry['det_conf'],
                                    ocr_conf=entry['ocr_conf'],
                                    source=entry.get('source', str(source_label)),
                                    source_type='live',
                                    num_reads=entry['num_reads'],
                                    time_sec=entry['time_sec'],
                                    is_valid=True,
                                )
                            except Exception as e:
                                print(f"  ⚠ DB write failed: {e}")
                        # POST to external API
                        if api_endpoint:
                            self._send_to_api(
                                plate_text, entry['det_conf'],
                                camera_id, api_endpoint)

            csv_file.close()

            # Summary
            print(f"\n{'='*60}")
            print(f"  SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"  Frames processed : {processed_count}")
            print(f"  Plates confirmed : {len(all_confirmed)}")
            for p in all_confirmed:
                print(f"    ✓ {p['plate']}  (reads={p['num_reads']})")
            print(f"  CSV saved → {csv_path}")

            if processed_count > 0:
                self._print_latency_report(processed_count)

        return all_confirmed

    # ------------------------------------------------------------------
    # Track merging (post-hoc)
    # ------------------------------------------------------------------
    def _merge_track_fragments(self, tracks: List[PlateTrack]) -> List[PlateTrack]:
        """
        Merge short track fragments that likely belong to the same vehicle/plate.
        This reduces duplicate outputs when bbox association is slightly off
        (common when plates are far/blurry).
        """
        if not tracks:
            return tracks

        merge_max_gap_frames = 30
        merge_dist_norm_thresh = 1.5  # Goldilocks zone: merges same fast car, doesn't eat other cars

        tracks_sorted = sorted(tracks, key=lambda t: t.start_frame_idx)
        merged: List[PlateTrack] = []

        for t in tracks_sorted:
            if not merged:
                merged.append(t)
                continue

            prev = merged[-1]
            gap = t.start_frame_idx - prev.last_frame_idx
            if gap < 0:
                gap = 0

            prev_center = _bbox_center(prev.last_bbox)
            # Use the first bbox of the later fragment for association.
            t_center = _bbox_center(t.bboxes[0])
            dist = float(np.sqrt((prev_center[0] - t_center[0]) ** 2 +
                                  (prev_center[1] - t_center[1]) ** 2))

            prev_diag = _bbox_diagonal(prev.last_bbox)
            dist_norm = dist / (prev_diag + 1e-6)

            # Merge if it's close in space and starts soon after the previous fragment.
            if gap <= merge_max_gap_frames and dist_norm <= merge_dist_norm_thresh:
                prev.bboxes.extend(t.bboxes)
                prev.reads.extend(t.reads)
                prev.last_frame_idx = t.last_frame_idx
            else:
                merged.append(t)

        return merged

    # ------------------------------------------------------------------
    # Detection (in-memory)
    # ------------------------------------------------------------------

    def _detect_in_memory(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO on a numpy frame.
        In fast mode: single pass only.
        In full mode: also tries CLAHE variant and merges.
        """
        dets1 = self._yolo_infer(frame)

        if self.fast_mode:
            return dets1  # Single pass — skip CLAHE + NMS overhead

        dets2 = self._yolo_infer(_enhance_clahe(frame))
        # Merge + NMS
        all_dets = dets1 + dets2
        return self._nms(all_dets)

    def _yolo_infer(self, img: np.ndarray) -> List[Dict[str, Any]]:
        return self.yolo.detect(img, conf=config.DETECTION_CONFIDENCE)

    def _nms(self, dets: List[dict], iou_thresh: float = 0.5) -> List[dict]:
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        keep = []
        while dets:
            cur = dets.pop(0)
            keep.append(cur)
            dets = [d for d in dets
                    if _calculate_iou(cur["bbox"], d["bbox"]) < iou_thresh]
        return keep

    # ------------------------------------------------------------------
    # Cropping (in-memory)
    # ------------------------------------------------------------------

    def _crop_plate(self, frame: np.ndarray, bbox: List[int],
                    padding: int = 12) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        pad = max(padding, config.SMALL_PLATE_PADDING) \
            if area < config.SMALL_PLATE_AREA_THRESHOLD else padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Track resolution (temporal voting)
    # ------------------------------------------------------------------

    def _resolve_tracks(self, tracks: List[PlateTrack],
                        video_name: str) -> List[Dict[str, Any]]:
        """Pick the best read from each track, deduplicate by plate text."""
        confirmed = []
        min_reads = getattr(config, "VIDEO_MIN_TRACK_READS", 2)
        min_plate_stability_ratio = getattr(config, "VIDEO_MIN_PLATE_STABILITY_RATIO", 0.6)
        min_det_conf = getattr(config, "VIDEO_MIN_DET_CONF", 0.0)
        min_ocr_conf = getattr(config, "VIDEO_MIN_OCR_CONF", 0.0)

        for track in tracks:
            best = track.best_read()
            if len(track.reads) < min_reads:
                print(f"[DEBUG] Dropped track (too few reads): {best.get('text', '')} - reads: {len(track.reads)}")
                continue

            # Only emit validated plate reads to avoid garbage strings.
            if not best.get("is_valid"):
                print(f"[DEBUG] Dropped track (invalid plate): {best.get('text', '')}")
                continue

            # Optional stability gate: only run if enabled (ratio > 0).
            if min_plate_stability_ratio > 0 and best.get("best_key_ratio", 0.0) < min_plate_stability_ratio:
                print(f"[DEBUG] Dropped track (unstable ratio): {best.get('text', '')} - ratio {best.get('best_key_ratio', 0.0)}")
                continue

            # Filter by YOLO detection confidence (reduces false positives early).
            if best.get("det_conf", 0.0) < min_det_conf:
                print(f"[DEBUG] Dropped track (low det conf): {best.get('text', '')}")
                continue

            # Filter by OCR confidence (reduces “valid but wrong” OCR).
            if best.get("ocr_conf", 0.0) < min_ocr_conf:
                print(f"[DEBUG] Dropped track (low ocr conf): {best.get('text', '')}")
                continue

            confirmed.append({
                "video": video_name,
                "plate": best["text"],
                "is_valid": best["is_valid"],
                "frame_index": best["frame_idx"],
                "time_sec": round(best["time_sec"], 2),
                "det_conf": round(best["det_conf"], 3),
                "ocr_conf": round(float(best.get("ocr_conf", 0.0)), 4),
                "num_reads": len(track.reads),
                "ocr_engine": self.ocr.engine_name,
            })

        return confirmed

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def _write_csv(self, rows: List[dict], csv_path: str):
        if not rows:
            print("\n  ⚠  No confirmed plates detected in video.")
            return
        fieldnames = list(rows[0].keys())

        # Write CSV (if enabled)
        if getattr(config, 'DB_ALSO_CSV', True):
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"\n  ✓ Plate results saved → {csv_path}")

        # Write to database
        if _DB_AVAILABLE and getattr(config, 'DB_ENABLED', True):
            try:
                count = anpr_db.insert_detections_batch(rows, source_type="video")
                print(f"  ✓ {count} plate(s) saved to database")
            except Exception as e:
                print(f"  ⚠ DB write failed: {e}")

        print(f"    ({len(rows)} unique plate(s) confirmed)")
        for r in rows:
            tag = "✓" if r["is_valid"] else "?"
            print(f"      {tag}  {r['plate']}  "
                  f"(conf={r['det_conf']}, reads={r['num_reads']}, "
                  f"t={r['time_sec']}s)")

    # ------------------------------------------------------------------
    # Latency report
    # ------------------------------------------------------------------

    def _print_latency_report(self, processed_count: int):
        print(f"\n{'='*60}")
        print(f"  LATENCY REPORT  ({processed_count} frames processed)")
        print(f"{'='*60}")

        for stage in ("detect", "crop", "ocr", "total"):
            vals = self.timings[stage]
            if not vals:
                continue
            vals_ms = [v * 1000 for v in vals]
            avg = statistics.mean(vals_ms)
            mn = min(vals_ms)
            mx = max(vals_ms)
            p95 = sorted(vals_ms)[int(len(vals_ms) * 0.95)] if len(vals_ms) > 1 else mx
            med = statistics.median(vals_ms)

            print(f"\n  {stage.upper():>8s}:")
            print(f"    Min    = {mn:8.2f} ms")
            print(f"    Avg    = {avg:8.2f} ms")
            print(f"    Median = {med:8.2f} ms")
            print(f"    P95    = {p95:8.2f} ms")
            print(f"    Max    = {mx:8.2f} ms")

        # Summary throughput
        if self.timings["total"]:
            avg_total = statistics.mean(self.timings["total"])
            est_fps = 1.0 / avg_total if avg_total > 0 else 0
            print(f"\n  Effective throughput: {est_fps:.1f} FPS "
                  f"(avg {avg_total*1000:.1f} ms/frame)")

        # Save JSON
        report_path = os.path.join(config.OUTPUT_DIR, "latency_report.json")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        report_data = {"frames_processed": processed_count}
        for stage in ("detect", "crop", "ocr", "total"):
            vals = self.timings[stage]
            if not vals:
                continue
            vals_ms = [v * 1000 for v in vals]
            p95_idx = int(len(vals_ms) * 0.95) if len(vals_ms) > 1 else len(vals_ms) - 1
            report_data[stage] = {
                "min_ms": round(min(vals_ms), 2),
                "avg_ms": round(statistics.mean(vals_ms), 2),
                "max_ms": round(max(vals_ms), 2),
                "p95_ms": round(sorted(vals_ms)[p95_idx], 2),
            }
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n  ✓ Latency report saved → {report_path}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Industry-Grade Video & Live ANPR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-recorded video
  python video_pipeline.py -i "test_videos/clip.mp4"
  python video_pipeline.py -i "test_videos/clip.mp4" --engine easyocr --frame-skip 5

  # Live camera (webcam)
  python video_pipeline.py --live --source 0

  # Live camera (IP / RTSP)
  python video_pipeline.py --live --source "rtsp://admin:pass@192.168.1.100/stream"
        """,
    )
    parser.add_argument("--input", "-i", default=None,
                        help="Path to video file (for offline mode)")
    parser.add_argument("--live", action="store_true",
                        help="Enable live camera mode")
    parser.add_argument("--source", default="0",
                        help="Camera source: webcam index (0,1,...) or "
                             "RTSP/HTTP URL (default: 0)")
    parser.add_argument("--engine", choices=["rapidocr", "paddle", "easyocr"],
                        default="rapidocr", help="OCR engine (default: rapidocr — PaddleOCR models via ONNX)")
    parser.add_argument("--frame-skip", type=int,
                        default=None, help="Process every Nth frame (default: from config)")
    parser.add_argument("--output-csv", default=None,
                        help="Custom CSV output path")
    parser.add_argument("--no-window", action="store_true",
                        help="Disable live display window (headless mode)")
    parser.add_argument("--camera-id", default="CAM_001",
                        help="Camera identifier for API payloads (default: CAM_001)")
    parser.add_argument("--api-endpoint", default=None,
                        help="URL to POST confirmed plates to "
                             "(e.g. http://93.127.172.217:2006/api/v1/scan)")
    args = parser.parse_args()

    if not args.live and not args.input:
        parser.error("Either --input or --live is required.")

    pipeline = VideoPipeline(engine=args.engine, frame_skip=args.frame_skip)

    if args.live:
        pipeline.process_live(
            source=args.source,
            csv_path=args.output_csv,
            show_window=not args.no_window,
            camera_id=args.camera_id,
            api_endpoint=args.api_endpoint,
        )
    else:
        pipeline.process(args.input, csv_path=args.output_csv)

    print("Done!")


if __name__ == "__main__":
    main()
