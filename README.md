# YOLOv11 Object Detection Project

This project is a Python-based application that performs object detection using YOLOv11 models. It converts PyTorch models to ONNX format and provides high-performance inference using ONNX Runtime.

## 🗂️ Directory Structure

```plaintext
yolov11l_test/
├── README.md
├── test_images/               # Test images folder
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── yolov11n/                  # YOLOv11 nano model
│   ├── export.py              # Model conversion script
│   ├── main.py                # Object detection execution script
│   ├── models/                # Model files storage folder
│   │   ├── yolo11l.pt         # PyTorch model file
│   │   └── yolo11l.onnx       # Converted ONNX model file
│   └── output/                # Detection result images storage folder
│       └── *_detected.jpg
└── (To be added in the future)
    ├── yolo11l-pose/          # YOLOv11 pose estimation
    ├── yolo11l-seg/           # YOLOv11 instance segmentation
    └── yolo11l-obb/           # YOLOv11 oriented bounding box
```

## 🛠️ Prerequisites

### 1. Python Environment Requirements

- Python 3.8 or higher

### 2. Required Package Installation

```bash
# Install basic packages
pip install ultralytics opencv-python numpy onnxruntime pathlib

# Or using requirements.txt
pip install -r requirements.txt
```

### 3. Main Dependencies

- **ultralytics**: YOLOv11 model loading and conversion
- **opencv-python**: Image processing and visualization
- **numpy**: Numerical computation
- **onnxruntime**: ONNX model inference
- **pathlib**: File path handling

## 📥 Model Download

### YOLOv11 Model Download

YOLOv11 models can be downloaded from [Ultralytics official documentation](https://docs.ultralytics.com/models/yolo11/#performance-metrics).

```bash
# Navigate to yolo11n folder
cd yolo11n

# Create models folder
mkdir models

# Download YOLOv11l model (approximately 50MB)
# Use one of the following methods:

# Method 1: Using wget
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt -O models/yolo11l.pt

# Method 2: Direct download
# Download https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt from browser and save to models/ folder
```

## 🚀 Usage

### 1. Model Conversion (PyTorch → ONNX)

```bash
cd yolo11n
python export.py
```

**export.py execution results:**

- Convert `models/yolo11l.pt` → `models/yolo11l.onnx`
- Automatic SSL certificate issue resolution
- Offline mode support
- Maximum compatibility using ONNX opset 12

### 2. Object Detection Execution

```bash
cd yolo11n
python main.py
```

**main.py execution process:**

1. Search for all `*.jpg`, `*.jpeg` files in `../test_images/` folder
2. Perform object detection on each image
3. Save detection results in `output/{filename}_detected.jpg` format
4. Output detected object information to console

### 3. Execution Result Example

```plaintext
Processing 9 image files in total.
Results will be saved in 'output' folder.
--------------------------------------------------

[1/9] Processing: 1.jpg
Detection result saved to 'output\1_detected.jpg' file.
[1] Total 8 objects detected.
  1. person: 0.91 - Position: (385, 69) ~ (499, 347)
  2. bowl: 0.81 - Position: (30, 343) ~ (99, 384)
  3. spoon: 0.76 - Position: (531, 43) ~ (553, 126)
  ...

--------------------------------------------------
Processing completed: 9/9 successful
Result files saved in 'output' folder.
```

## 📊 Supported Object Classes

YOLOv11 is trained on the COCO dataset and can detect 80 classes of objects:

**People and Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Daily Items**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture and Electronics**: chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

**Others**: backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## ⚙️ Configuration Options

### main.py Configuration

```python
# Model and path settings
ONNX_MODEL_PATH = 'models/yolo11l.onnx'  # ONNX model path
IMAGE_PATH = '../test_images/'            # Input image path (file or folder)
OUTPUT_DIR = 'output'                     # Result storage folder

# Detection threshold settings
CONFIDENCE_THRESHOLD = 0.5    # Confidence threshold (0.0 ~ 1.0)
IOU_THRESHOLD = 0.5          # IoU threshold (for duplicate removal)
SCORE_THRESHOLD = 0.5        # Score threshold
```

## 🔧 Troubleshooting

### SSL Certificate Error

- SSL certificate errors may occur in corporate firewall or proxy environments
- `export.py` includes SSL bypass code that automatically resolves these issues

### ONNX Runtime Error

```bash
# Reinstall CPU version
pip uninstall onnxruntime
pip install onnxruntime

# For GPU version (requires NVIDIA GPU)
pip install onnxruntime-gpu
```

### Memory Shortage Error

- Memory shortage may occur when processing large images or many images
- Reduce batch size or resize images before processing

## 📈 Performance Information

- **Model Size**: YOLOv11l (approximately 50MB)
- **Inference Speed**: approximately 200-500ms per image on CPU
- **Supported Resolution**: up to 640x640 (automatic resizing)
- **Accuracy**: mAP50-95: 53.9% (based on COCO val2017)

## 🚧 Future Plans

- **yolo11l-pose/**: Add pose estimation functionality
- **yolo11l-seg/**: Add instance segmentation functionality
- **yolo11l-obb/**: Add oriented bounding box detection
- Add web interface
- Real-time video processing functionality
- Mobile-optimized version

## 📝 License

This project follows the MIT license. YOLOv11 models follow the [Ultralytics](https://ultralytics.com/) license.

## 🤝 Contributing

Bug reports, feature suggestions, and pull requests are welcome!

---

### Created with ❤️ using YOLOv11 and ONNX Runtime
