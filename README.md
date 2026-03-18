# Indian License Plate Recognition System

A high-accuracy Automatic Number Plate Recognition (ANPR) system optimized for Indian license plates. This project combines YOLOv8 for detection and a custom OCR pipeline leveraging DeepSeek OCR-2 and PaddleOCR.

## Features

- **Robust Detection**: Uses YOLOv8 to detect license plates, even in challenging conditions (night, angles, distance).
- **Advanced OCR**: Integrating DeepSeek OCR-2 and PaddleOCR with custom Indian plate formatting logic.
- **Intelligent Correction**: Features a specialized `IndianPlateFormatter` that corrects common OCR errors based on Indian state codes and plate formats (e.g., `KA 01` vs `KA O1`).
- **Edge Simulation**: Includes a simulation environment for testing on edge devices.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VedantGupta-DTU/ANPR.git
    cd ANPR
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Weights:**
    Due to GitHub file size limits, the trained model weights are hosted separately.
    
    *   **[Download best.pt (YOLOv8 Weights)](https://drive.google.com/file/d/1atAcgSSs3l1JLnFYbwA9-fYeTq2XbuKi/view?usp=sharing)** - Place this file in the root directory.
    

## Usage

### Running the Pipeline

To run the full recognition pipeline on a folder of images:

```bash
python pipeline.py -i test_images/ --engine paddle
```

### Options

-   `-i`, `--input`: Path to input image or directory.
-   `--engine`: OCR engine to use (`paddle`, `deepseek`, or `easyocr`).
-   `--conf`: Confidence threshold for detection (default: 0.25).

## Project Structure

-   `pipeline.py`: Main entry point for the ANPR system.
-   `plate_detector.py`: YOLOv8 wrapper for plate detection.
-   `ocr_reader.py`: Handles OCR processing using various engines.
-   `indian_plate_formatter.py`: Contains logic for validating and formatting Indian license plates.
-   `config.py`: Configuration settings.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
