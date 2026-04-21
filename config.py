"""
Configuration settings for License Plate Recognition Pipeline
"""
import os

# Performance mode: True = optimized for speed (10+ FPS), False = max accuracy
FAST_MODE = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
MODEL_PATH_ONNX = os.path.join(BASE_DIR, "best.onnx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

# YOLO inference input size
YOLO_IMGSZ = 640

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# YOLO Detection settings
DETECTION_CONFIDENCE = 0.50  # Balanced threshold — catches brief plates while filtering noise
IOU_THRESHOLD = 0.45         # IoU threshold for NMS

# Bike / small plate settings
SMALL_PLATE_AREA_THRESHOLD = 15000  # px² – plates below this get extra padding
SMALL_PLATE_PADDING = 25            # Extra pixels around small/bike plates
IGNORE_BIKE_PLATES = False           # Skip square/tall plates (bikes) and only detect cars

# DeepSeek OCR-2 settings
OCR_MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
OCR_PROMPT = "<image>\nFree OCR. "  # Prompt for clean text extraction
USE_FLASH_ATTENTION = True  # Set to False if flash-attn not installed

# Processing settings
DEVICE = "cuda"  # Use "cpu" if no GPU available
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Video settings
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
VIDEO_FRAME_SKIP = 2  # Process every Nth frame for speed

# Tracking / confirmation settings (video_pipeline.py)
# Only output a plate if its track is stable across multiple sampled frames.
VIDEO_MIN_TRACK_READS = 1

# Require the best plate text to dominate OCR reads for that track.
# This helps avoid “valid but noisy” OCR where different valid plates appear
# across frames for the same vehicle.
# Set to 0 to effectively disable the stability gate.
VIDEO_MIN_PLATE_STABILITY_RATIO = 0.0  # best_key_freq / pool_len

# Filter confirmed tracks by YOLO detection confidence (reduces false positives).
VIDEO_MIN_DET_CONF = 0.4

# For each plate track, only vote using the last portion of reads (later
# frames are usually less blurry because the car is closer).
VIDEO_OCR_TAIL_RATIO = 0.6

# Prefer truly confident OCR when multiple valid-looking plates compete.
# (PaddleOCR rec_scores are typically in [0,1].)
VIDEO_MIN_OCR_CONF = 0.45

# OCR confidence weighting power in track voting.
VIDEO_OCR_CONF_POWER = 3.0
VIDEO_LATENCY_LOG = os.path.join(OUTPUT_DIR, "latency_report.json")

# Edge / Raspberry Pi settings
MAX_PENDING_OCR = 5  # Drop new OCR tasks when queue exceeds this (prevents lag on Pi)

# Database settings (MongoDB Atlas — cloud database for Pi deployment)
MONGO_URI = "mongodb+srv://guptavedant2005_db_user:anpr12345678@cluster0.l29wepu.mongodb.net/?appName=Cluster0"
MONGO_DB_NAME = "anpr_system"                           # Database name
MONGO_COLLECTION = "plate_detections"                    # Collection name
DB_ENABLED = True                                        # Toggle database logging
DB_ALSO_CSV = True                                       # Keep CSV logging alongside DB

