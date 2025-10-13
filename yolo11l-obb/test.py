import os
from ultralytics import YOLO
import numpy as np

# Set offline mode
os.environ['YOLO_OFFLINE'] = '1'
os.environ['ULTRALYTICS_OFFLINE'] = '1'

# Try exactly matching our parameters
model = YOLO("models/yolo11l-obb.pt", task="obb")
results = model(source="../assets/boats.jpg", conf=0.25, iou=0.45, save=True, imgsz=1024)

for result in results:
    if result.obb is not None:
        print(f"Total OBB detections: {len(result.obb)}")
        print(f"OBB tensor shape: {result.obb.data.shape}")
        
        # Get confidence values
        confidences = result.obb.conf.cpu().numpy()
        print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
        print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
        
        # Check class distribution
        classes = result.obb.cls.cpu().numpy()
        unique_classes, counts = np.unique(classes, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
        
        # More detailed conf analysis
        conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(len(conf_bins)-1):
            count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
            print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
        print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
        
    else:
        print("No OBB detections found")
    
    # reference https://docs.ultralytics.com/modes/predict/ for more information.