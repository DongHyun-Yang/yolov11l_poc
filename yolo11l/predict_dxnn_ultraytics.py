# IMPORTANT: Import this BEFORE any ultralytics imports
import custom_ultralytics_setup

import cv2
import numpy as np
from ultralytics import YOLO

# Load the exported ONNX model
onnx_model = YOLO("yolo11l/models/yolo11l.dxnn")

# Run inference
results = onnx_model(source="assets/bus.jpg", save=True)

# # Access the results
# for result in results:
#     xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
#     xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
#     names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
#     confs = result.obb.conf  # confidence score of each box

#     # Visualization: draw OBBs on the image
#     img_vis = result.orig_img.copy()
#     for i in range(len(xywhr)):
#         poly = xyxyxyxy[i].cpu().numpy().astype(int)
#         conf = confs[i].item()
#         name = names[i]
#         # Draw polygon
#         cv2.polylines(img_vis, [poly.reshape(-1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)
#         # Draw label
#         label = f"{name} {conf:.2f}"
#         cv2.putText(img_vis, label, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#     # Show the image with detections
#     cv2.imshow("OBB Detections", img_vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     if result.obb is not None:
#         print("="*50)
#         print(f"Total OBB detections: {len(result.obb)}")
#         print(f"OBB tensor shape: {result.obb.data.shape}")
        
#         # Get confidence values
#         confidences = result.obb.conf.cpu().numpy()
#         print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
#         print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
        
#         # Check class distribution
#         classes = result.obb.cls.cpu().numpy()
#         unique_classes, counts = np.unique(classes, return_counts=True)
#         print(f"Class distribution: {dict(zip(unique_classes.astype(int), counts))}")
        
#         # More detailed conf analysis
#         conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#         for i in range(len(conf_bins)-1):
#             count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
#             print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
#         print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
        
#     else:
#         print("No OBB detections found")
