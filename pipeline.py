"""
License Plate Recognition Pipeline
Combines YOLO detection with OCR for end-to-end plate recognition
Supports: DeepSeek OCR-2 (GPU) → PaddleOCR (CPU) → EasyOCR (CPU fallback)
Now also supports video input with CSV export.
"""
import os
import cv2
import csv
import json
import torch
from datetime import datetime
from typing import List, Optional, Dict, Any
from plate_detector import PlateDetector
from ocr_reader import OCRReader, PaddleOCRReader, EasyOCRReader, EnsembleOCRReader
from indian_plate_formatter import IndianPlateFormatter
import config


class LicensePlateRecognizer:
    """End-to-end license plate recognition pipeline"""
    
    def __init__(self, ocr_engine: str = "auto", device: str = None):
        """
        Initialize the recognition pipeline
        
        Args:
            ocr_engine: OCR engine to use:
                'auto'     - DeepSeek if GPU, else PaddleOCR
                'paddle'   - PaddleOCR (best CPU accuracy)
                'easyocr'  - EasyOCR (lighter weight)
                'ensemble' - PaddleOCR + EasyOCR (picks best)
                'deepseek' - DeepSeek OCR-2 (requires GPU)
            device: Device for inference ('cuda' or 'cpu')
        """
        print("=" * 50)
        print("Initializing License Plate Recognition Pipeline")
        print("=" * 50)
        
        self.detector = PlateDetector()
        
        # Resolve 'auto' engine choice
        if ocr_engine == "auto":
            if torch.cuda.is_available():
                ocr_engine = "deepseek"
                print("\n[INFO] CUDA GPU detected → using DeepSeek OCR-2")
            else:
                ocr_engine = "paddle"
                print("\n[INFO] No CUDA GPU → using PaddleOCR (high accuracy, CPU)")
        
        # Initialize chosen OCR engine
        engines = {
            "paddle": lambda: PaddleOCRReader(),
            "easyocr": lambda: EasyOCRReader(),
            "ensemble": lambda: EnsembleOCRReader(),
            "deepseek": lambda: OCRReader(device=device),
        }
        
        self.ocr_engine_name = ocr_engine
        self.ocr = engines[ocr_engine]()
        print(f"Using OCR engine: {ocr_engine}")
        
        # Lazy load OCR model
        self._ocr_loaded = False
        self._formatter = IndianPlateFormatter()
    
    def _ensure_ocr_loaded(self):
        """Lazy load OCR model when first needed"""
        if not self._ocr_loaded:
            self.ocr.load_model()
            self._ocr_loaded = True
    
    def process_image(self, image_path: str, save_crops: bool = True, 
                      save_visualization: bool = True) -> List[dict]:
        """
        Process a single image through the full pipeline
        
        Args:
            image_path: Path to the input image
            save_crops: Save cropped plate images
            save_visualization: Save image with detection boxes
            
        Returns:
            List of results with plate info and OCR text
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*50}")
        
        results = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Step 1: Detect and crop plates
        print("\n[1/2] Detecting license plates...")
        crops = self.detector.crop_plates(image_path)
        print(f"      Found {len(crops)} plate(s)")
        
        if len(crops) == 0:
            print("      No plates detected!")
            return results
        
        # Step 2: OCR on each crop
        print("\n[2/2] Extracting text with OCR...")
        self._ensure_ocr_loaded()
        
        crop_dir = os.path.join(config.OUTPUT_DIR, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        
        for i, (crop_img, detection) in enumerate(crops):
            # Save crop temporarily (or permanently if requested)
            crop_path = os.path.join(crop_dir, f"{base_name}_plate_{i}.jpg")
            cv2.imwrite(crop_path, crop_img)
            
            # Run OCR
            try:
                text = self.ocr.read_plate(crop_path)
                print(f"      Plate {i+1}: {text} (conf: {detection['confidence']:.2f})")
            except Exception as e:
                text = f"OCR Error: {str(e)}"
                print(f"      Plate {i+1}: Error - {e}")
            
            results.append({
                'plate_index': i,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'text': text,
                'crop_path': crop_path if save_crops else None
            })
            
            if not save_crops:
                os.remove(crop_path)
        
        # Save visualization with annotations
        if save_visualization:
            self._save_annotated_image(image_path, results)
        
        return results
    
    def _save_annotated_image(self, image_path: str, results: List[dict]):
        """Save image with bounding boxes and OCR text"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            text = r['text']
            conf = r['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw text background
            label = f"{text} ({conf:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(image, (x1, y2), (x1 + label_w + 10, y2 + label_h + 15), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(image, label, (x1 + 5, y2 + label_h + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Save
        vis_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, os.path.basename(image_path))
        cv2.imwrite(vis_path, image)
        print(f"\n      Visualization saved: {vis_path}")
    
    def process_directory(self, input_dir: str) -> dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing images
            
        Returns:
            Dictionary mapping image paths to results
        """
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Not a directory: {input_dir}")
        
        # Find all images
        images = [f for f in os.listdir(input_dir) 
                  if os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS]
        
        print(f"\nFound {len(images)} images to process")
        
        all_results = {}
        for img_name in images:
            img_path = os.path.join(input_dir, img_name)
            try:
                results = self.process_image(img_path)
                all_results[img_path] = results
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                all_results[img_path] = {'error': str(e)}
        
        # Save summary
        self._save_summary(all_results)
        
        return all_results
    
    def _save_summary(self, all_results: dict):
        """Save a JSON summary of all results"""
        summary_path = os.path.join(config.OUTPUT_DIR, "results_summary.json")
        
        # Prepare JSON-serializable results
        json_results = {}
        for path, results in all_results.items():
            if isinstance(results, dict) and 'error' in results:
                json_results[path] = results
            else:
                json_results[path] = [
                    {k: v for k, v in r.items()} 
                    for r in results
                ]
        
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': json_results
            }, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Results summary saved: {summary_path}")
        print(f"{'='*50}")

    def process_video(
        self,
        video_path: str,
        frame_skip: int = None,
        csv_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a video, running plate detection + OCR on sampled frames.
        Only writes a plate to CSV once it passes format validation, to avoid
        noisy reads when the car is far/blurry.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        frame_skip = frame_skip or getattr(config, "VIDEO_FRAME_SKIP", 2)

        print(f"\n{'='*50}")
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"{'='*50}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        # Temporary frame directory
        tmp_dir = os.path.join(config.OUTPUT_DIR, "video_frames")
        os.makedirs(tmp_dir, exist_ok=True)

        # Track unique, high-quality plates we have already emitted
        seen_plates: Dict[str, Dict[str, Any]] = {}
        rows: List[Dict[str, Any]] = []

        frame_idx = 0
        video_name = os.path.basename(video_path)

        # Ensure OCR is loaded once
        self._ensure_ocr_loaded()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            timestamp_sec = frame_idx / fps

            # Save current frame as an image and reuse the existing image pipeline
            frame_filename = f"{os.path.splitext(video_name)[0]}_f{frame_idx:06d}.jpg"
            frame_path = os.path.join(tmp_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            try:
                # Reuse existing plate detection + OCR logic
                results = self.process_image(
                    frame_path,
                    save_crops=True,
                    save_visualization=False,
                )
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                frame_idx += 1
                continue

            for r in results:
                crop_path = r.get("crop_path")
                det_conf = r.get("confidence", 0.0)
                bbox = r.get("bbox", [0, 0, 0, 0])

                if not crop_path or not os.path.exists(crop_path):
                    continue

                # Use validation-aware OCR if available to avoid low-confidence garbage
                plate_text = ""
                is_valid = False

                if hasattr(self.ocr, "read_plate_with_validation"):
                    try:
                        val = self.ocr.read_plate_with_validation(crop_path)
                        plate_text = (val.get("formatted") or val.get("raw") or "").strip().upper()
                        is_valid = bool(val.get("is_valid"))
                    except Exception as e:
                        print(f"Validation OCR error on frame {frame_idx}: {e}")
                        continue
                else:
                    # Fallback: validate the already-formatted text using the Indian formatter
                    raw_text = (r.get("text") or "").strip().upper()
                    is_valid, formatted = self._formatter.validate_plate(raw_text)
                    plate_text = (formatted if is_valid else raw_text).strip().upper()

                if not is_valid:
                    # Skip unreadable / low-quality / invalid-format plates
                    continue

                key = plate_text.replace(" ", "")
                if key in seen_plates:
                    # We already have a good read for this plate; no need to spam CSV
                    continue

                record = {
                    "video": video_name,
                    "plate": plate_text,
                    "frame_index": frame_idx,
                    "time_sec": round(timestamp_sec, 2),
                    "det_conf": float(f"{det_conf:.3f}"),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "ocr_engine": self.ocr_engine_name,
                }
                seen_plates[key] = record
                rows.append(record)

            frame_idx += 1

        cap.release()

        # Clean up temporary frames to avoid clutter
        try:
            import shutil

            shutil.rmtree(tmp_dir)
        except Exception:
            # Best-effort cleanup; ignore errors
            pass

        if not rows:
            print("\nNo high-confidence, valid plates detected in video.")
            return []

        if csv_path is None:
            base = os.path.splitext(video_name)[0]
            csv_path = os.path.join(config.OUTPUT_DIR, f"{base}_plates.csv")

        fieldnames = [
            "video",
            "plate",
            "frame_index",
            "time_sec",
            "det_conf",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "ocr_engine",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n{'='*50}")
        print(f"Video results saved: {csv_path}")
        print(f"{'='*50}")

        return rows


def main():
    """CLI for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="License Plate Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image (auto-selects best engine)
  python pipeline.py -i test_images/car.jpg
  
  # Process all images with PaddleOCR
  python pipeline.py -i test_images/ --engine paddle
  
  # Use ensemble mode (PaddleOCR + EasyOCR, picks best)
  python pipeline.py -i test_images/ --engine ensemble
        """
    )
    
    parser.add_argument("--input", "-i", required=True, 
                       help="Input image or directory")
    parser.add_argument("--output", "-o", default=config.OUTPUT_DIR,
                       help="Output directory")
    parser.add_argument("--engine", choices=["auto", "paddle", "easyocr", "ensemble", "deepseek"],
                       default="auto", help="OCR engine (default: auto)")
    parser.add_argument("--use-easyocr", action="store_true",
                       help="Shortcut for --engine easyocr")
    parser.add_argument("--no-crops", action="store_true",
                       help="Don't save cropped plate images")
    parser.add_argument("--no-vis", action="store_true",
                       help="Don't save visualizations")
    
    args = parser.parse_args()
    
    # Handle legacy flag
    engine = "easyocr" if args.use_easyocr else args.engine
    
    # Override output directory
    config.OUTPUT_DIR = args.output
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize pipeline
    pipeline = LicensePlateRecognizer(ocr_engine=engine)
    
    # Process
    if os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[1].lower()
        if ext in getattr(config, "IMAGE_EXTENSIONS", []):
            results = pipeline.process_image(
                args.input,
                save_crops=not args.no_crops,
                save_visualization=not args.no_vis
            )
            
            print("\n" + "="*50)
            print("RESULTS")
            print("="*50)
            for r in results:
                print(f"  Plate: {r['text']}")
                print(f"  Confidence: {r['confidence']:.2f}")
                print(f"  Bounding Box: {r['bbox']}")
                print()
        elif ext in getattr(config, "VIDEO_EXTENSIONS", []):
            # Video input: run frame-by-frame processing and export CSV
            pipeline.process_video(args.input)
        else:
            raise ValueError(f"Unsupported input file type: {ext}")
    else:
        pipeline.process_directory(args.input)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
