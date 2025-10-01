import os
import ssl
import urllib3
from ultralytics import YOLO

# Disable SSL verification (to resolve SSL issues due to corporate firewalls, etc.)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set offline mode
os.environ['YOLO_OFFLINE'] = '1'
os.environ['ULTRALYTICS_OFFLINE'] = '1'

# Load YOLOv11l-pose model
model = YOLO('models/yolo11l-pose.pt')

# Convert model to ONNX format (opset=12 specified for general compatibility)
model.export(format='onnx', opset=12)

print("Successfully converted YOLOv11l-pose model to ONNX format. 'models/yolo11l-pose.onnx' file has been created.")