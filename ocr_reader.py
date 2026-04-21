"""
OCR Reader Module - Multi-Engine Support
Supports DeepSeek OCR-2, PaddleOCR, and EasyOCR with automatic fallback
Includes image preprocessing for improved license plate accuracy
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List
import config
from indian_plate_formatter import IndianPlateFormatter
import re


# ─────────────────────────────────────────────────────────────────────
# Image Preprocessing for License Plates
# ─────────────────────────────────────────────────────────────────────

def preprocess_plate_image(image_path: str, save_debug: bool = False) -> List[np.ndarray]:
    """
    Apply multiple preprocessing techniques to improve OCR accuracy.
    Returns a list of preprocessed images (original + variants).
    Handles both single-line (car) and two-line (bike) plates.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    results = []
    h, w = img.shape[:2]
    aspect_ratio = h / w if w > 0 else 0
    is_bike_plate = aspect_ratio > 0.7  # Tall/square = likely 2-line bike plate
    
    # --- 1. Aggressive resize for small plates ---
    if w < 300 or h < 80:
        # Bike plates need even more upscaling
        if is_bike_plate:
            scale = max(300 / w, 150 / h, 3.0)
        else:
            scale = max(300 / w, 80 / h, 2.5)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    
    results.append(img)  # original (possibly resized)
    
    # --- 2. Grayscale + CLAHE (adaptive contrast) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    results.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
    
    # --- 3. Sharpened ---
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    results.append(sharpened)
    
    # --- 4. Adaptive threshold (binary) ---
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    results.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    
    # --- 5. Otsu threshold ---
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    
    # --- 6. Morphological denoising (helps small/noisy plates) ---
    denoised = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    results.append(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))
    
    # --- 7. Bike plate: horizontal split (top half + bottom half) ---
    # Indian bike plates are typically 2 lines: "MH 12" on top, "AB 1234" on bottom
    if is_bike_plate:
        mid_y = h // 2
        # Give a generous overlap band (25% of height) so characters at the boundary aren't clipped (e.g. D -> L)
        overlap = max(int(h * 0.25), 8)
        top_half = img[0 : mid_y + overlap, :]
        bottom_half = img[mid_y - overlap : h, :]
        results.append(top_half)
        results.append(bottom_half)
        
        # Also try CLAHE on each half
        top_gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
        bot_gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        results.append(cv2.cvtColor(clahe.apply(top_gray), cv2.COLOR_GRAY2BGR))
        results.append(cv2.cvtColor(clahe.apply(bot_gray), cv2.COLOR_GRAY2BGR))
        
        # Also try sharpened halves (helps with E/F/5 confusion)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        results.append(cv2.filter2D(top_half, -1, kernel))
        results.append(cv2.filter2D(bottom_half, -1, kernel))
    
    if save_debug:
        debug_dir = os.path.join(config.OUTPUT_DIR, "debug_preprocess")
        os.makedirs(debug_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        for i, r in enumerate(results):
            cv2.imwrite(os.path.join(debug_dir, f"{base}_variant_{i}.jpg"), r)
    
    return results


def is_bike_plate_image(image_path: str) -> bool:
    """Check if a plate crop looks like a 2-line bike plate (tall aspect ratio)."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    return (h / w) > 0.7 if w > 0 else False


# ─────────────────────────────────────────────────────────────────────
# PaddleOCR Reader (recommended — highest accuracy on CPU)
# ─────────────────────────────────────────────────────────────────────

class PaddleOCRReader:
    """High-accuracy OCR using PaddleOCR 3.4+"""
    
    def __init__(self, format_indian: bool = True):
        self.ocr = None
        self.format_indian = format_indian
        self.indian_formatter = IndianPlateFormatter() if format_indian else None
    
    def load_model(self):
        if self.ocr is not None:
            return
        import os
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        from paddleocr import PaddleOCR
        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(lang='en')
        print("PaddleOCR loaded!")
    
    def _ocr_image(self, img) -> str:
        """Run OCR on a numpy image (BGR) or image path.
        Sorts text boxes top-to-bottom by Y position and filters hologram noise.
        """
        result = self.ocr.predict(img)
        
        for res in result:
            if 'dt_polys' in res and 'rec_texts' in res:
                # Sort by Y position (top-to-bottom reading order)
                entries = []
                for poly, text, score in zip(res['dt_polys'], res['rec_texts'], res.get('rec_scores', [1.0] * len(res['rec_texts']))):
                    min_y = min(p[1] for p in poly)
                    min_x = min(p[0] for p in poly)
                    text_upper = text.upper().strip()
                    # Filter out hologram / emblem noise
                    if text_upper in ('IND', 'INDIA', 'IN', 'ONI', 'ND', 'INO'):
                        continue
                    # Skip very low confidence junk
                    if score < 0.3:
                        continue
                    entries.append((min_y, min_x, text_upper))
                
                # Sort by Y first (top to bottom), then X (left to right)
                entries.sort(key=lambda e: (e[0], e[1]))
                texts = [e[2] for e in entries]
                return ' '.join(texts).upper() if texts else ""
            
            # Fallback: no positional data
            if 'rec_texts' in res:
                texts = [t.upper().strip() for t in res['rec_texts']
                         if t.upper().strip() not in ('IND', 'INDIA', 'IN', 'ONI', 'ND', 'INO')]
                return ' '.join(texts).upper() if texts else ""
        
        return ""
    
    def read_plate(self, image_path: str) -> str:
        if self.ocr is None:
            self.load_model()
        
        bike_plate = is_bike_plate_image(image_path)
        
        # Try multiple preprocessed versions and pick the best
        variants = preprocess_plate_image(image_path)
        
        best_text = ""
        best_score = -1
        
        # For bike plates, also try concatenating top + bottom half OCR results
        half_texts = []  # collect OCR from half-image variants
        
        for idx, variant in enumerate(variants):
            raw = self._ocr_image(variant)
            if not raw:
                continue
            
            # Track half-image results for bike plates
            # Halves are appended after the 6 full-image variants (indices 6-9)
            if bike_plate and idx >= 6:
                half_texts.append(raw.strip())
            
            if self.indian_formatter:
                is_valid, formatted = self.indian_formatter.validate_plate(raw)
                
                clean_fmt = formatted.replace(' ', '')
                clean_raw = re.sub(r'[^A-Z0-9]', '', raw.upper())
                diff = abs(len(clean_raw) - len(clean_fmt))
                
                # Penalize long invalid strings (likely stickers/slogans)
                if not is_valid and len(clean_fmt) > 15:
                    score = -100
                else:
                    score = (100 + len(clean_fmt) - (diff * 2)) if is_valid else len(clean_fmt)
                
                # Character-level similarity bonus: prefer variants where the
                # raw OCR closely matches the formatted plate (fewer corrections).
                # This breaks ties between e.g. "CAQ" (read correctly) vs
                # "CA0" (read as 0, corrected to O → CAO).
                if is_valid and len(clean_raw) == len(clean_fmt):
                    matching = sum(1 for a, b in zip(clean_raw, clean_fmt) if a == b)
                    score += matching  # +1 per matching char position
                
                if score > best_score:
                    best_score = score
                    best_text = formatted
            else:
                # If no formatter, just pick longest? 
                # But filter out clearly too long junk (stickers)
                if len(raw) > len(best_text) and len(raw) < 16:
                    best_text = raw
        
        # For bike plates: try combining pairs of half-text results
        # Pairs are: (6,7)=Raw, (8,9)=CLAHE, (10,11)=Sharpened
        if bike_plate and len(half_texts) >= 2 and self.indian_formatter:
            for i in range(0, len(half_texts) - 1, 2):
                combined = half_texts[i] + ' ' + half_texts[i + 1]
                is_valid, formatted = self.indian_formatter.validate_plate(combined)
                
                clean_fmt = formatted.replace(' ', '')
                clean_raw = re.sub(r'[^A-Z0-9]', '', combined.upper())
                diff = abs(len(clean_raw) - len(clean_fmt))
                
                # Base score: favor longer valid plates (full matches > partial matches)
                score = (100 + len(clean_fmt) - (diff * 2)) if is_valid else len(clean_fmt)
                
                if score > best_score:
                    best_score = score
                    best_text = formatted
        
        return best_text if best_text else "UNREADABLE"
    
    def read_plate_with_validation(self, image_path: str) -> dict:
        if self.ocr is None:
            self.load_model()
        
        bike_plate = is_bike_plate_image(image_path)
        variants = preprocess_plate_image(image_path)
        
        best = {'raw': '', 'formatted': '', 'is_valid': False, 'score': -1}
        half_texts = []
        
        for idx, variant in enumerate(variants):
            raw = self._ocr_image(variant)
            if not raw:
                continue
            
            if bike_plate and idx >= 6:
                half_texts.append(raw.strip())
            
            if self.indian_formatter:
                is_valid, formatted = self.indian_formatter.validate_plate(raw)
                
                clean_fmt = formatted.replace(' ', '')
                clean_raw = re.sub(r'[^A-Z0-9]', '', raw.upper())
                diff = abs(len(clean_raw) - len(clean_fmt))
                
                score = (100 + len(clean_fmt) - (diff * 2)) if is_valid else len(clean_fmt)
                
                if score > best['score']:
                    best = {'raw': raw, 'formatted': formatted, 'is_valid': is_valid, 'score': score}
        
            if self.indian_formatter:
                for i in range(0, len(half_texts) - 1, 2):
                    combined = half_texts[i] + ' ' + half_texts[i + 1]
                    is_valid, formatted = self.indian_formatter.validate_plate(combined)
                    score = (100 + len(formatted.replace(' ', ''))) if is_valid else len(formatted.replace(' ', ''))
                    if score > best['score']:
                        best = {'raw': combined, 'formatted': formatted, 'is_valid': is_valid, 'score': score}
        
        del best['score']
        return best


# ─────────────────────────────────────────────────────────────────────
# EasyOCR Reader (lighter weight fallback)
# ─────────────────────────────────────────────────────────────────────

class EasyOCRReader:
    """Fallback OCR using EasyOCR"""
    
    def __init__(self, languages: List[str] = None, format_indian: bool = True):
        self.languages = languages or ['en']
        self.reader = None
        self.format_indian = format_indian
        self.indian_formatter = IndianPlateFormatter() if format_indian else None
    
    def load_model(self):
        if self.reader is not None:
            return
        import easyocr
        import torch
        print(f"Loading EasyOCR for languages: {self.languages}")
        self.reader = easyocr.Reader(self.languages, gpu=torch.cuda.is_available())
        print("EasyOCR loaded!")
    
    def read_plate(self, image_path: str) -> str:
        if self.reader is None:
            self.load_model()
        
        bike_plate = is_bike_plate_image(image_path)
        variants = preprocess_plate_image(image_path)
        
        best_text = ""
        best_score = -1
        half_texts = []
        
        for idx, variant in enumerate(variants):
            # EasyOCR expects a numpy array
            results = self.reader.readtext(variant)
            texts = [r[1] for r in results]
            raw = ' '.join(texts).upper()
            if not raw:
                continue
            
            if bike_plate and idx >= 6:
                half_texts.append(raw.strip())
            
            if self.format_indian and self.indian_formatter:
                is_valid, formatted = self.indian_formatter.validate_plate(raw)
                
                clean_fmt = formatted.replace(' ', '')
                clean_raw = re.sub(r'[^A-Z0-9]', '', raw.upper())
                diff = abs(len(clean_raw) - len(clean_fmt))
                
                score = (100 + len(clean_fmt) - (diff * 2)) if is_valid else len(clean_fmt)
                
                # Character-level similarity bonus: prefer variants where raw matches formatted
                if is_valid and len(clean_raw) == len(clean_fmt):
                    matching = sum(1 for a, b in zip(clean_raw, clean_fmt) if a == b)
                    score += matching
                
                if score > best_score:
                    best_score = score
                    best_text = formatted
            else:
                if len(raw) > len(best_text):
                    best_text = raw
        
        if bike_plate and len(half_texts) >= 2 and self.indian_formatter:
            for i in range(0, len(half_texts) - 1, 2):
                combined = half_texts[i] + ' ' + half_texts[i + 1]
                is_valid, formatted = self.indian_formatter.validate_plate(combined)
                
                clean_fmt = formatted.replace(' ', '')
                clean_raw = re.sub(r'[^A-Z0-9]', '', combined.upper())
                diff = abs(len(clean_raw) - len(clean_fmt))
                
                score = (100 + len(clean_fmt) - (diff * 2)) if is_valid else len(clean_fmt)
                
                if score > best_score:
                    best_score = score
                    best_text = formatted
        
        return best_text if best_text else "UNREADABLE"
    
    def read_plate_with_validation(self, image_path: str) -> dict:
        if self.reader is None:
            self.load_model()
        results = self.reader.readtext(image_path)
        texts = [r[1] for r in results]
        raw_text = ' '.join(texts).upper()
        if self.indian_formatter:
            is_valid, formatted = self.indian_formatter.validate_plate(raw_text)
            return {'raw': raw_text, 'formatted': formatted, 'is_valid': is_valid}
        return {'raw': raw_text, 'formatted': raw_text, 'is_valid': False}


# ─────────────────────────────────────────────────────────────────────
# Multi-Engine Ensemble (tries PaddleOCR + EasyOCR, picks best)
# ─────────────────────────────────────────────────────────────────────

class EnsembleOCRReader:
    """Runs multiple OCR engines and picks the best result"""
    
    def __init__(self, format_indian: bool = True):
        self.format_indian = format_indian
        self.indian_formatter = IndianPlateFormatter() if format_indian else None
        self.paddle = PaddleOCRReader(format_indian=format_indian)
        self.easyocr = EasyOCRReader(format_indian=format_indian)
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return
        self.paddle.load_model()
        self.easyocr.load_model()
        self._loaded = True
    
    def read_plate(self, image_path: str) -> str:
        if not self._loaded:
            self.load_model()
        
        # Get results from both engines
        paddle_result = self.paddle.read_plate_with_validation(image_path)
        easy_result = self.easyocr.read_plate_with_validation(image_path)
        
        # Pick the best result
        # Priority: valid Indian plate > longer text
        if paddle_result.get('is_valid') and not easy_result.get('is_valid'):
            return paddle_result['formatted']
        elif easy_result.get('is_valid') and not paddle_result.get('is_valid'):
            return easy_result['formatted']
        elif paddle_result.get('is_valid') and easy_result.get('is_valid'):
            # Both valid — prefer PaddleOCR (generally more accurate)
            return paddle_result['formatted']
        else:
            # Neither valid — pick the one with more alphanumeric chars
            p_len = len(paddle_result.get('formatted', '').replace(' ', ''))
            e_len = len(easy_result.get('formatted', '').replace(' ', ''))
            
            # Filter out long garbage (stickers)
            if p_len > 15 and e_len > 15:
                return "UNREADABLE"
            
            if p_len >= e_len:
                return paddle_result.get('formatted', 'UNREADABLE')
            return easy_result.get('formatted', 'UNREADABLE')
    
    def read_plate_with_validation(self, image_path: str) -> dict:
        if not self._loaded:
            self.load_model()
        
        paddle_result = self.paddle.read_plate_with_validation(image_path)
        easy_result = self.easyocr.read_plate_with_validation(image_path)
        
        # Same selection logic
        if paddle_result.get('is_valid'):
            return {**paddle_result, 'engine': 'paddle'}
        elif easy_result.get('is_valid'):
            return {**easy_result, 'engine': 'easyocr'}
        else:
            p_len = len(paddle_result.get('formatted', '').replace(' ', ''))
            e_len = len(easy_result.get('formatted', '').replace(' ', ''))
            if p_len >= e_len:
                return {**paddle_result, 'engine': 'paddle'}
            return {**easy_result, 'engine': 'easyocr'}


# ─────────────────────────────────────────────────────────────────────
# DeepSeek OCR-2 Reader (GPU only)
# ─────────────────────────────────────────────────────────────────────

class OCRReader:
    """DeepSeek OCR-2 based text reader (requires CUDA GPU)"""
    
    def __init__(self, model_name: str = None, device: str = None):
        import torch
        self.model_name = model_name or config.OCR_MODEL_NAME
        self.device = device or config.DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return
        print(f"Loading DeepSeek OCR-2 from {self.model_name}...")
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        attn = 'flash_attention_2' if config.USE_FLASH_ATTENTION else 'eager'
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            self.model = AutoModel.from_pretrained(
                self.model_name, _attn_implementation=attn,
                trust_remote_code=True, use_safetensors=True)
        except Exception:
            from transformers import AutoModel
            import torch
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True, use_safetensors=True)
        self.model = self.model.eval()
        if self.device == "cuda":
            self.model = self.model.cuda().to(torch.bfloat16)
        self._loaded = True
        print("DeepSeek OCR-2 loaded!")
    
    def read_plate(self, image_path: str) -> str:
        if not self._loaded:
            self.load_model()
        result = self.model.infer(
            self.tokenizer, prompt="<image>\nFree OCR. ",
            image_file=image_path, output_path=None,
            base_size=1024, image_size=768,
            crop_mode=True, save_results=False)
        return str(result).strip().upper() if result else ""


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="License Plate OCR")
    parser.add_argument("--input", "-i", required=True, help="Image or directory")
    parser.add_argument("--output", "-o", default=None, help="Output file")
    parser.add_argument("--engine", choices=["paddle", "easyocr", "ensemble", "deepseek"],
                       default="paddle", help="OCR engine (default: paddle)")
    args = parser.parse_args()
    
    engines = {
        "paddle": PaddleOCRReader,
        "easyocr": EasyOCRReader,
        "ensemble": EnsembleOCRReader,
        "deepseek": OCRReader,
    }
    reader = engines[args.engine]()
    
    if os.path.isfile(args.input):
        images = [args.input]
    else:
        images = [os.path.join(args.input, f) for f in os.listdir(args.input)
                  if os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS]
    
    print(f"\nProcessing {len(images)} images with {args.engine}...")
    results = []
    for image_path in images:
        print(f"\n--- {image_path} ---")
        try:
            text = reader.read_plate(image_path)
            print(f"  Result: {text}")
            results.append({'path': image_path, 'text': text})
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'path': image_path, 'text': '', 'error': str(e)})
    
    if args.output:
        with open(args.output, 'w') as f:
            for r in results:
                f.write(f"{r['path']}: {r['text']}\n")
        print(f"\nSaved to: {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
