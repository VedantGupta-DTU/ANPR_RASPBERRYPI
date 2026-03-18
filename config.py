"""
Configuration settings for License Plate Recognition Pipeline
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# YOLO Detection settings
DETECTION_CONFIDENCE = 0.35  # Minimum confidence for plate detection (lowered for bike plates)
IOU_THRESHOLD = 0.45         # IoU threshold for NMS

# Bike / small plate settings
SMALL_PLATE_AREA_THRESHOLD = 15000  # px² – plates below this get extra padding
SMALL_PLATE_PADDING = 15            # Extra pixels around small/bike plates (default is 5)

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
