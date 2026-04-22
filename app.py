"""
ANPR Web Application
====================
Flask-based web frontend for the ANPR pipeline.
Supports: Live Demo | Video Upload | Image Upload

Usage:
    python app.py
    # Open http://localhost:5000
"""

import os
import sys
import uuid
import json
import time
import csv
import datetime
import tempfile
import threading
from typing import Optional

import cv2
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   Response, send_from_directory)

import config
import db as anpr_db
from indian_plate_formatter import IndianPlateFormatter

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

UPLOAD_DIR = os.path.join(config.OUTPUT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton pipeline
# ---------------------------------------------------------------------------

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    """Load the pipeline once and reuse it across requests."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from video_pipeline import VideoPipeline
                _pipeline = VideoPipeline(engine="rapidocr", frame_skip=3)
    return _pipeline


# ---------------------------------------------------------------------------
# Live camera state
# ---------------------------------------------------------------------------

_live_active = False
_live_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Routes: Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes: Image Upload API
# ---------------------------------------------------------------------------

@app.route("/api/process-image", methods=["POST"])
def process_image():
    """Accept an image file, run detection + OCR, return JSON results."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported image format: {ext}"}), 400

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(tmp_path)

    try:
        pipeline = get_pipeline()
        t0 = time.perf_counter()

        # Read image
        frame = cv2.imread(tmp_path)
        if frame is None:
            return jsonify({"error": "Cannot read image"}), 400

        # Detect
        detections = pipeline._detect_in_memory(frame)

        results = []
        for det in detections:
            crop = pipeline._crop_plate(frame, det["bbox"])

            # Filter bikes
            if getattr(config, "IGNORE_BIKE_PLATES", False):
                crop_h, crop_w = crop.shape[:2]
                if crop_w > 0 and (crop_h / crop_w) > 0.65:
                    continue

            text, is_valid, ocr_conf = pipeline.ocr.read_numpy(crop)
            results.append({
                "plate": text,
                "is_valid": is_valid,
                "det_conf": round(det["confidence"], 3),
                "ocr_conf": round(ocr_conf, 3),
                "bbox": det["bbox"],
            })

        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        # Draw annotated image for display
        annotated = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color = (0, 255, 0) if r["is_valid"] else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if r["is_valid"]:
                label = f"{r['plate']} ({r['det_conf']:.2f})"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10),
                              (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Save annotated image
        ann_path = os.path.join(UPLOAD_DIR, f"annotated_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(ann_path, annotated)
        ann_filename = os.path.basename(ann_path)

        # Save valid plates to database
        for r in results:
            if r["is_valid"]:
                anpr_db.insert_detection(
                    plate=r["plate"],
                    det_conf=r["det_conf"],
                    ocr_conf=r["ocr_conf"],
                    source="image_upload",
                    source_type="image",
                    is_valid=r["is_valid"],
                    bbox=r["bbox"],
                )

        return jsonify({
            "plates": [r for r in results if r["is_valid"]],
            "all_detections": results,
            "elapsed_ms": elapsed,
            "annotated_image": f"/uploads/{ann_filename}",
        })

    finally:
        # Cleanup uploaded file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Routes: Video Upload API
# ---------------------------------------------------------------------------

@app.route("/api/process-video", methods=["POST"])
def process_video():
    """Accept a video file, run the full pipeline, return JSON results."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported video format: {ext}"}), 400

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(tmp_path)

    try:
        pipeline = get_pipeline()
        t0 = time.perf_counter()

        csv_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_plates.csv")
        confirmed = pipeline.process(tmp_path, csv_path=csv_path)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        plates = []
        for c in confirmed:
            plates.append({
                "plate": c.get("plate", ""),
                "det_conf": c.get("det_conf", 0),
                "ocr_conf": c.get("ocr_conf", 0),
                "num_reads": c.get("num_reads", 0),
                "time_sec": c.get("time_sec", 0),
            })

        # Save to database (video pipeline already writes CSV;
        # DB insert handled in video_pipeline._write_csv)

        return jsonify({
            "plates": plates,
            "elapsed_ms": elapsed,
            "frame_count": len(pipeline.timings.get("total", [])),
        })

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)


# ---------------------------------------------------------------------------
# Routes: Live Camera Feed (MJPEG)
# ---------------------------------------------------------------------------

def _generate_live_frames(source=0):
    """Generator: yield MJPEG frames with ANPR overlay.

    Architecture (async producer-consumer):
      Main thread:  capture → detect → draw bbox → encode JPEG → yield
      OCR thread:   pull crops from queue → run OCR → update shared results

    Result: bounding boxes appear instantly (~15ms with TensorRT),
    plate text fills in asynchronously (~0.5-1s later).
    Video feed NEVER freezes waiting for OCR.
    """
    import threading
    import queue as queue_mod

    global _live_active
    pipeline = get_pipeline()

    # ── Open camera: try GStreamer (Jetson CSI) → V4L2/USB fallback ──
    cap = None
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cam_id = int(source) if isinstance(source, str) else source
        # Try Jetson GStreamer pipeline first (for CSI cameras like IMX219)
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={cam_id} ! "
            f"video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[CAM] Opened Jetson CSI camera via GStreamer")
        else:
            cap.release()
            # Fallback: standard USB / V4L2
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[CAM] Opened USB camera {cam_id}")
    else:
        # URL or RTSP stream
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print("[CAM] ERROR: Could not open any camera!")
        yield b''
        return

    frame_idx = 0
    frame_skip = pipeline.frame_skip

    # ── Shared state between main thread and OCR worker ──
    # Key = bbox tuple, Value = {"text": str, "conf": float, "is_valid": bool}
    ocr_results = {}
    ocr_results_lock = threading.Lock()
    ocr_queue = queue_mod.Queue(maxsize=getattr(config, 'MAX_PENDING_OCR', 5))
    ocr_stop = threading.Event()

    # Track which bbox regions we've already sent for OCR (dedup)
    _ocr_cache = {}  # key=bbox_key → True (already queued)

    # Prepare CSV file
    csv_path = "result.csv"
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(["timestamp", "plate", "det_conf", "ocr_conf"])

    def _bbox_key(bbox):
        """Round bbox to 20px grid for dedup (same plate = same grid cell)."""
        return (bbox[0] // 20, bbox[1] // 20, bbox[2] // 20, bbox[3] // 20)

    # ── OCR Worker Thread ──
    def _ocr_worker():
        """Background thread: processes plate crops and updates shared results."""
        while not ocr_stop.is_set():
            try:
                item = ocr_queue.get(timeout=0.5)
            except queue_mod.Empty:
                continue

            crop, bbox, det_conf = item
            bbox_k = _bbox_key(bbox)

            try:
                text, is_valid, ocr_conf = pipeline.ocr.read_numpy(crop)
            except Exception:
                text, is_valid, ocr_conf = "", False, 0.0

            # Update shared results
            with ocr_results_lock:
                ocr_results[bbox_k] = {
                    "text": text if is_valid else "",
                    "conf": det_conf,
                    "ocr_conf": ocr_conf,
                    "is_valid": is_valid,
                    "bbox": bbox,
                }

            # Log valid plates
            if is_valid:
                ts = datetime.datetime.now().isoformat()
                try:
                    if getattr(config, 'DB_ALSO_CSV', True):
                        csv_writer.writerow([
                            ts, text,
                            round(det_conf, 3),
                            round(ocr_conf, 3)
                        ])
                        csv_file.flush()
                    anpr_db.insert_detection(
                        plate=text,
                        det_conf=round(det_conf, 3),
                        ocr_conf=round(ocr_conf, 3),
                        source="live_camera",
                        source_type="live",
                        is_valid=True,
                        bbox=bbox,
                        timestamp=ts,
                    )
                except Exception:
                    pass

            ocr_queue.task_done()

    # Start OCR worker
    ocr_thread = threading.Thread(target=_ocr_worker, daemon=True)
    ocr_thread.start()

    # ── Current detection bboxes (updated every detection frame) ──
    current_detections = []  # List of {"bbox", "conf"}

    with _live_lock:
        _live_active = True

    try:
        retry_count = 0
        while _live_active:
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                if retry_count > 50:
                    break
                time.sleep(0.1)
                continue

            retry_count = 0

            # ── Edge optimization: cap frame resolution to 640px wide ──
            max_w = 640
            h_orig, w_orig = frame.shape[:2]
            if w_orig > max_w:
                scale = max_w / w_orig
                frame = cv2.resize(frame, (max_w, int(h_orig * scale)),
                                   interpolation=cv2.INTER_AREA)

            # ── Run detection on sampled frames (FAST — GPU accelerated) ──
            if frame_idx % frame_skip == 0:
                detections = pipeline._detect_in_memory(frame)
                current_detections = []

                # Clear stale OCR cache entries
                active_keys = set()

                for det in detections:
                    bbox = det["bbox"]
                    bbox_k = _bbox_key(bbox)
                    active_keys.add(bbox_k)

                    current_detections.append({
                        "bbox": bbox,
                        "conf": det["confidence"],
                    })

                    # Skip bike plates if configured
                    if getattr(config, "IGNORE_BIKE_PLATES", False):
                        crop = pipeline._crop_plate(frame, bbox)
                        crop_h, crop_w = crop.shape[:2]
                        if crop_w > 0 and (crop_h / crop_w) > 0.65:
                            continue

                    # Only queue OCR if we haven't already for this bbox position
                    if bbox_k not in _ocr_cache:
                        _ocr_cache[bbox_k] = True
                        crop = pipeline._crop_plate(frame, bbox)
                        try:
                            ocr_queue.put_nowait(
                                (crop, bbox, det["confidence"]))
                        except queue_mod.Full:
                            pass  # Drop — OCR is overloaded, skip this plate

                # Prune stale cache entries (plates that left the frame)
                stale_keys = [k for k in _ocr_cache if k not in active_keys]
                for k in stale_keys:
                    _ocr_cache.pop(k, None)
                    with ocr_results_lock:
                        ocr_results.pop(k, None)

            # ── Draw bounding boxes (instant — no OCR wait) ──
            display = frame.copy()
            snapshot_needed = False

            for det in current_detections:
                bbox = det["bbox"]
                bbox_k = _bbox_key(bbox)
                x1, y1, x2, y2 = bbox

                # Check if OCR result is available for this bbox
                with ocr_results_lock:
                    ocr_res = ocr_results.get(bbox_k)

                if ocr_res and ocr_res.get("text"):
                    # Green = recognized plate
                    color = (0, 255, 0)
                    label = f"{ocr_res['text']} ({det['conf']:.2f})"
                    snapshot_needed = True
                else:
                    # Cyan = detected, pending OCR
                    color = (255, 255, 0)
                    label = f"Detecting... ({det['conf']:.2f})"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display, (x1, y1 - th - 10),
                              (x1 + tw, y1), color, -1)
                cv2.putText(display, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Save snapshot if valid plates detected
            if snapshot_needed and frame_idx % frame_skip == 0:
                snapshot_path = os.path.join(
                    UPLOAD_DIR, f"live_plate_{uuid.uuid4().hex[:8]}.jpg")
                cv2.imwrite(snapshot_path, display)

            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', display,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')

            frame_idx += 1

    finally:
        ocr_stop.set()
        ocr_thread.join(timeout=2)
        cap.release()
        try:
            csv_file.close()
        except Exception:
            pass
        with _live_lock:
            _live_active = False


@app.route("/api/live-feed")
def live_feed():
    """MJPEG stream endpoint for the live camera feed."""
    source_arg = request.args.get("source", "0")
    try:
        source = int(source_arg)
    except ValueError:
        source = source_arg

    return Response(
        _generate_live_frames(source),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/live-stop", methods=["POST"])
def live_stop():
    """Stop the live camera feed."""
    global _live_active
    with _live_lock:
        _live_active = False
    return jsonify({"status": "stopped"})


# ---------------------------------------------------------------------------
# Routes: Serve uploaded/annotated files
# ---------------------------------------------------------------------------

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ---------------------------------------------------------------------------
# Routes: Database Query APIs
# ---------------------------------------------------------------------------

@app.route("/api/plates")
def api_plates():
    """Query plate detections with optional filters.
    
    Query params:
        plate   - substring search (e.g. ?plate=DL+9)
        source  - exact source match
        type    - source_type filter (live/video/image)
        since   - ISO timestamp start
        until   - ISO timestamp end
        min_det - minimum detection confidence
        min_ocr - minimum OCR confidence
        limit   - max results (default 100)
        offset  - pagination offset
    """
    try:
        results = anpr_db.query_plates(
            plate_filter=request.args.get("plate"),
            start_time=request.args.get("since"),
            end_time=request.args.get("until"),
            source=request.args.get("source"),
            source_type=request.args.get("type"),
            min_det_conf=float(request.args["min_det"]) if "min_det" in request.args else None,
            min_ocr_conf=float(request.args["min_ocr"]) if "min_ocr" in request.args else None,
            limit=int(request.args.get("limit", 100)),
            offset=int(request.args.get("offset", 0)),
        )
        return jsonify({"count": len(results), "plates": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Get detection statistics summary."""
    try:
        stats = anpr_db.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recent-plates")
def api_recent_plates():
    """Get most recent plate detections."""
    limit = int(request.args.get("limit", 20))
    try:
        results = anpr_db.get_recent_plates(limit=limit)
        return jsonify({"count": len(results), "plates": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export-csv")
def api_export_csv():
    """Export filtered detections as a CSV download."""
    import io

    try:
        results = anpr_db.query_plates(
            plate_filter=request.args.get("plate"),
            start_time=request.args.get("since"),
            end_time=request.args.get("until"),
            source=request.args.get("source"),
            source_type=request.args.get("type"),
            limit=int(request.args.get("limit", 99999)),
        )

        if not results:
            return jsonify({"error": "No records found"}), 404

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=anpr_export.csv"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ANPR Web Application")
    print("=" * 60)

    # Initialize database
    try:
        anpr_db.init_db()
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}. Running without DB.")

    # Pre-load models so the first request is fast
    get_pipeline()

    print("  Models loaded!")
    print(f"  Open http://localhost:5005 in your browser")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5005, debug=False, threaded=True)
