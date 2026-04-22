#!/usr/bin/env python3
"""
Convert YOLOv8 ONNX → TFLite  (TinyML)
========================================
Converts best.onnx to best.tflite for ultra-fast ARM inference.

Usage (run on any machine with pip — NOT on Jetson):
    pip install ultralytics
    python convert_to_tflite.py

Or on Google Colab (recommended):
    !pip install ultralytics
    !python convert_to_tflite.py

This requires the original .pt model OR the ultralytics package.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_with_ultralytics():
    """Convert using the ultralytics export (most reliable method)."""
    from ultralytics import YOLO

    pt_path = os.path.join(BASE_DIR, "best.pt")
    if not os.path.exists(pt_path):
        print(f"ERROR: {pt_path} not found.")
        print("Upload your best.pt model to this directory first.")
        sys.exit(1)

    print(f"Loading {pt_path} ...")
    model = YOLO(pt_path)

    # Export FP32 TFLite (works everywhere)
    print("\n=== Exporting FP32 TFLite ===")
    model.export(
        format="tflite",
        imgsz=320,       # Match config.YOLO_IMGSZ for edge
    )

    # Also export INT8 quantized (half the size, ~2x faster on ARM)
    print("\n=== Exporting INT8 Quantized TFLite ===")
    try:
        model.export(
            format="tflite",
            imgsz=320,
            int8=True,
        )
        print("INT8 export succeeded!")
    except Exception as e:
        print(f"INT8 export failed (non-critical): {e}")
        print("FP32 model will still work fine.")

    print("\n✓ Done! Copy the .tflite file to your Jetson/Pi.")
    print("  Look for: best_saved_model/best_float32.tflite")
    print("  Or:       best_saved_model/best_int8.tflite")


def convert_with_onnx2tf():
    """Fallback: convert ONNX directly using onnx2tf."""
    try:
        import onnx2tf
    except ImportError:
        print("Installing onnx2tf ...")
        os.system(f"{sys.executable} -m pip install onnx2tf onnx tensorflow")
        import onnx2tf

    onnx_path = os.path.join(BASE_DIR, "best.onnx")
    if not os.path.exists(onnx_path):
        print(f"ERROR: {onnx_path} not found.")
        sys.exit(1)

    out_dir = os.path.join(BASE_DIR, "best_tflite")
    print(f"Converting {onnx_path} → TFLite ...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        non_verbose=True,
    )
    print(f"\n✓ Done! TFLite model saved to: {out_dir}/")


if __name__ == "__main__":
    pt_path = os.path.join(BASE_DIR, "best.pt")

    if os.path.exists(pt_path):
        print("Found best.pt — using ultralytics export (recommended)")
        convert_with_ultralytics()
    else:
        print("No best.pt found — using onnx2tf conversion")
        convert_with_onnx2tf()
