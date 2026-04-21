# ANPR Pipeline — Comprehensive Documentation

> **Automatic Number Plate Recognition** system for Indian license plates using YOLO v8 detection and PaddleOCR / EasyOCR text recognition.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Reference](#2-module-reference)
3. [Configuration Reference](#3-configuration-reference)
4. [OCR Correction Logic](#4-ocr-correction-logic)
5. [Usage Guide](#5-usage-guide)
6. [Performance & Optimization](#6-performance--optimization)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT SOURCE                                │
│   Video File (.mp4)  │  Webcam (index 0)  │  RTSP / IP Camera   │
└──────────┬───────────┴────────┬───────────┴──────────┬──────────┘
           │                    │                      │
           ▼                    ▼                      ▼
    ┌──────────────────────────────────────────────────────────┐
    │              cv2.VideoCapture (frame reader)             │
    │         Frame skip: every Nth frame processed            │
    └──────────────────────┬───────────────────────────────────┘
                           │  numpy BGR frame
                           ▼
    ┌──────────────────────────────────────────────────────────┐
    │          STAGE 1: Plate Detection (YOLO v26m)            │
    │                                                          │
    │   • Original frame → YOLO inference                      │
    │   • CLAHE-enhanced frame → YOLO inference                │
    │   • Merge + Non-Max Suppression (NMS)                    │
    │   • Output: list of bounding boxes + confidence scores   │
    └──────────────────────┬───────────────────────────────────┘
                           │  [x1, y1, x2, y2] + conf
                           ▼
    ┌──────────────────────────────────────────────────────────┐
    │          STAGE 2: Plate Cropping                         │
    │                                                          │
    │   • Extract plate region with adaptive padding           │
    │   • Bike plate filter (aspect ratio > 0.65 = skip)       │
    └──────────────────────┬───────────────────────────────────┘
                           │  cropped numpy image
                           ▼
    ┌──────────────────────────────────────────────────────────┐
    │          STAGE 3: OCR (PaddleOCR / EasyOCR)              │
    │                                                          │
    │   • Multiple preprocessing variants (grayscale, CLAHE,   │
    │     binary, inverted, sharpened)                          │
    │   • Candidate ranking with directional weights           │
    │   • IndianPlateFormatter strict positional corrections    │
    │   • Output: plate text + is_valid + confidence           │
    └──────────────────────┬───────────────────────────────────┘
                           │  "DL 12 D 4547", True, 0.95
                           ▼
    ┌──────────────────────────────────────────────────────────┐
    │          STAGE 4: Temporal Tracking & Voting              │
    │                                                          │
    │   • IoU-based bounding box association across frames     │
    │   • Weighted consensus voting (det_conf × ocr_conf³)     │
    │   • Track merging for fragmented detections              │
    │   • Minimum reads / stability / confidence gates         │
    └──────────────────────┬───────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────────┐
    │          OUTPUT                                           │
    │   • CSV file with confirmed plates                       │
    │   • Console summary with confidence & read counts        │
    │   • Latency report (JSON)                                │
    │   • Live: bounding box overlay window + real-time emit   │
    └──────────────────────────────────────────────────────────┘
```

---

## 2. Module Reference

### `video_pipeline.py` — Main Pipeline (Video + Live)

The primary entry point for both pre-recorded video and live camera ANPR.

| Class / Function | Purpose |
|---|---|
| `VideoPipeline` | Main orchestrator. Loads YOLO + OCR, manages tracks, timing. |
| `VideoPipeline.process(video_path)` | Offline mode — processes a video file end-to-end. |
| `VideoPipeline.process_live(source)` | **Live mode** — connects to webcam/IP/RTSP, shows real-time overlay. |
| `PlateTrack` | Accumulates OCR reads for a single tracked plate across frames. |
| `PlateTrack.best_read()` | Temporal voting: picks the best plate text using weighted consensus. |
| `InMemoryOCR` | Wraps PaddleOCR / EasyOCR for in-memory numpy inference. |
| `InMemoryOCR.read_numpy(crop)` | Runs OCR on a cropped plate image with multi-variant preprocessing. |
| `_calculate_iou()` | IoU between two bounding boxes (track association). |
| `_enhance_clahe()` | Applies CLAHE to a frame for improved contrast. |

### `indian_plate_formatter.py` — Plate Formatting & Validation

| Class / Function | Purpose |
|---|---|
| `IndianPlateFormatter` | Cleans, corrects, formats, and validates OCR output. |
| `format_plate(text)` | Main entry: clean → correct → extract components → format. |
| `validate_plate(text)` | Checks if text matches Indian plate regex patterns. |
| `_apply_ocr_corrections(text)` | **Strict positional mapping** (see Section 4). |
| `extract_components(text)` | Splits into state, district, series, number. |
| `INDIAN_STATE_CODES` | List of all valid Indian state codes. |

### `config.py` — Global Configuration

All tunable parameters in one place. See Section 3 for full reference.

### `plate_detector.py` — YOLO Detection Wrapper

| Class / Function | Purpose |
|---|---|
| `PlateDetector` | Wraps YOLO for plate detection on images. |
| `detect(image_path)` | Detects plates in a single image file. |
| `crop_plates(image_path)` | Detects + crops plates, returns cropped images. |

### `ocr_reader.py` — Multi-Engine OCR Reader

| Class / Function | Purpose |
|---|---|
| `OCRReader` | Supports DeepSeek OCR-2, PaddleOCR, EasyOCR, Ensemble. |
| `read_plate(image_path)` | Full pipeline: preprocess → OCR → validate. |
| `preprocess_plate_image(img)` | Specialized preprocessing for license plate crops. |

### `pipeline.py` — Legacy Pipeline (Single Image / Directory)

| Class / Function | Purpose |
|---|---|
| `LicensePlateRecognizer` | Original pipeline for images/dirs/videos. |
| `process_image(path)` | Detects + reads plates in a single image. |
| `process_directory(dir)` | Batch processes all images in a folder. |
| `process_video(path)` | Legacy video processing (disk-based, no tracking). |

### `benchmark_onnx.py` — Model Benchmarking

Measures inference latency for YOLO in `.pt` vs `.onnx` format.

---

## 3. Configuration Reference

All settings are in `config.py`:

### Paths

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `best.pt` | Path to YOLO model weights |
| `OUTPUT_DIR` | `output/` | Directory for CSV, JSON, crop outputs |
| `TEST_IMAGES_DIR` | `test_images/` | Directory for test images |

### Detection

| Parameter | Default | Description |
|---|---|---|
| `DETECTION_CONFIDENCE` | `0.85` | Minimum YOLO confidence to accept a detection |
| `IOU_THRESHOLD` | `0.45` | IoU threshold for Non-Max Suppression |
| `IGNORE_BIKE_PLATES` | `True` | Skip tall/square plates (bikes) — car-only mode |
| `SMALL_PLATE_AREA_THRESHOLD` | `15000` | Plates below this area (px²) get extra padding |
| `SMALL_PLATE_PADDING` | `15` | Extra crop padding for small plates |

### Video / Live Pipeline

| Parameter | Default | Description |
|---|---|---|
| `VIDEO_FRAME_SKIP` | `2` | Process every Nth frame (higher = faster, less accurate) |
| `VIDEO_MIN_TRACK_READS` | `2` | Min OCR reads before a track is confirmed |
| `VIDEO_MIN_PLATE_STABILITY_RATIO` | `0.0` | Disabled. Min ratio of best plate text frequency. |
| `VIDEO_MIN_DET_CONF` | `0.5` | Min YOLO confidence to confirm a plate track |
| `VIDEO_MIN_OCR_CONF` | `0.45` | Min OCR confidence to confirm a plate track |
| `VIDEO_OCR_CONF_POWER` | `3.0` | Cubic weighting — high-conf reads dominate voting |
| `VIDEO_OCR_TAIL_RATIO` | `0.6` | *(Legacy, now disabled)* Fraction of tail reads to use |

### OCR

| Parameter | Default | Description |
|---|---|---|
| `OCR_MODEL_NAME` | `deepseek-ai/DeepSeek-OCR-2` | DeepSeek OCR model name |
| `DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |

---

## 4. OCR Correction Logic

### Strict Positional Character Enforcement

The `_apply_ocr_corrections()` function in `indian_plate_formatter.py` enforces rigid character-type rules based on position:

```
Position:     [0] [1] [2] [3] [4...N-5] [N-5] [N-4] [N-3] [N-2] [N-1]
Must be:       L   L   D   D*   L        L     D     D     D     D

L = Letter (A-Z)
D = Digit (0-9)
D* = Digit if OCR output looks like a digit (4th char heuristic)
```

**Rules applied in order:**

1. **Positions 0–1** (State Code): Force digits to letters (`0→O`, `1→I`, `5→S`, `8→B`, etc.)
2. **Position 2** (District digit 1): Force letters to digits (`O→0`, `I→1`, `S→5`, etc.)
3. **Position 3** (District digit 2): If it resembles a digit, map to digit; if it resembles a letter, map to letter.
4. **Positions 4 to N-5** (Series): Force digits to letters.
5. **Position N-5** (5th from last): Force digits to letters.
6. **Positions N-4 to N-1** (Number): Force letters to digits (`D→0`, `O→0`, `B→8`, etc.)

### Confusion Maps

| Digit → Letter | Letter → Digit |
|---|---|
| `0 → O` | `O → 0` |
| `1 → I` | `I → 1`, `L → 1` |
| `2 → Z` | `Z → 2` |
| `4 → A` | `A → 4` |
| `5 → S` | `S → 5` |
| `6 → G` | `G → 6` |
| `7 → T` | `T → 7` |
| `8 → B` | `B → 8`, `D → 0`, `Q → 0` |

### Directional Weights (video_pipeline.py)

When `InMemoryOCR` ranks candidates from multiple preprocessing variants, it uses directional weights to bias corrections:

| OCR Raw → Candidate | Weight | Effect |
|---|---|---|
| `O → D` | `1.10` | Strongly favors D over raw O |
| `N → W` | `1.10` | Strongly favors W over raw N |
| `0 → D` | `0.95` | Slightly favors D over exact 0 |
| `0 → O` | `0.85` | Slightly favors O over exact 0 |

---

## 5. Usage Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/VedantGupta-DTU/DTU-SECURITY.git
cd "deepseek ocr-2"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Pre-Recorded Video Mode

Process a video file with temporal tracking and plate voting:

```bash
python video_pipeline.py --input "test_videos/clip.mp4"

# Options:
#   --engine paddle|easyocr    OCR backend (default: paddle)
#   --frame-skip N             Process every Nth frame (default: from config)
#   --output-csv path.csv      Custom CSV output path
```

**Output:**
- `output/<video_name>_plates.csv` — Confirmed plates
- `output/latency_report.json` — Per-stage timing stats
- Console summary with plate list and latency breakdown

### Live Camera Mode

Connect a webcam, IP camera, or RTSP stream for real-time detection:

```bash
# Webcam (default camera)
python video_pipeline.py --live --source 0

# IP camera / RTSP stream
python video_pipeline.py --live --source "rtsp://admin:pass@192.168.1.100/stream"

# HTTP stream
python video_pipeline.py --live --source "http://192.168.1.100:8080/video"

# Headless mode (no display window — e.g., for server deployments)
python video_pipeline.py --live --source 0 --no-window
```

**Controls:**
- Press `q` in the live window to stop.
- `Ctrl+C` also works for graceful shutdown.

**Live window features:**
- Green bounding boxes around detected plates with OCR text overlay
- Red bounding boxes for detected-but-unreadable plates
- FPS counter and confirmed plate count in top-left corner
- Plates emitted to console in real-time as vehicles leave the frame

**Output:**
- `output/live_plates.csv` — Continuously appended as plates are detected
- Console summary on exit with all confirmed plates and latency breakdown

### Single Image Mode

```bash
python pipeline.py --input test_images/plate.jpg --engine paddle
```

### Batch Directory Mode

```bash
python pipeline.py --input test_images/ --engine paddle
```

---

## 6. Performance & Optimization

### Latency Benchmarks (CPU — Apple M-series)

| Stage | Avg | P95 | Max |
|---|---|---|---|
| **YOLO Detection** | 200 ms | 250 ms | 430 ms |
| **Plate Cropping** | < 1 ms | < 1 ms | < 1 ms |
| **PaddleOCR** | 175 ms | 1930 ms | 2250 ms |
| **Total (per frame)** | 375 ms | 2130 ms | 2450 ms |

**Effective throughput:** ~2.7 FPS on CPU (frames with no plates are faster).

### Optimization Tips

1. **GPU Acceleration**: Use CUDA-enabled PaddleOCR and YOLO for 10-20× speedup. Set `config.DEVICE = "cuda"`.
2. **ONNX Runtime**: Export YOLO to ONNX (`best.onnx`) for 30-50% faster CPU inference:
   ```bash
   python export_model.py
   ```
3. **Frame Skip**: Increase `VIDEO_FRAME_SKIP` for faster processing at the cost of detection coverage.
4. **Reduce Detection Confidence**: Lower `DETECTION_CONFIDENCE` to catch more plates (may increase false positives).
5. **Edge Deployment**: Use Jetson Nano / Hailo-8 NPU for real-time (25+ FPS) on embedded devices.

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **In-memory processing** | Eliminates disk I/O bottleneck (~10× faster than disk-based pipeline) |
| **Dual YOLO inference** (original + CLAHE) | Catches plates in both well-lit and low-light conditions |
| **Temporal voting** | Eliminates single-frame OCR noise; consensus across 2+ frames |
| **Cubic confidence weighting** | Prevents blurry garbage frames from outvoting clear reads |
| **Strict positional char mapping** | Forces OCR output to structurally match Indian plate format |
| **IoU-based track association** | Lightweight alternative to SORT/DeepSORT for single-camera setups |

---

*Generated for the DTU Security ANPR Project.*
