#!/usr/bin/env python3
"""
Compare pose estimation results between ultralytics and our custom implementation
Both using ONNX models for fair performance comparison
"""

import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import time
import sys
import os

# Import functions from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import letterbox, postprocess_pose

PT_MODEL_PATH = 'models/yolo11l-pose.pt'
ONNX_MODEL_PATH = 'models/yolo11l-pose.onnx'
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
DEVICE = 'cpu'  # Force CPU for consistency
TEST_IMAGE_PATH = '../assets/bus.jpg'

# COCO pose keypoints
POSE_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton connections for pose visualization (matching ultralytics exactly)
POSE_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9), (8, 10),
    (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)
]

# COCO pose colors (matching ultralytics pose_palette)
POSE_PALETTE = [
    [255, 128, 0],    # 0: Orange
    [255, 153, 51],   # 1: Light Orange
    [255, 178, 102],  # 2: Light Orange
    [230, 230, 0],    # 3: Yellow
    [255, 153, 255],  # 4: Pink
    [153, 204, 255],  # 5: Light Blue
    [255, 102, 255],  # 6: Magenta
    [255, 51, 255],   # 7: Bright Magenta
    [102, 178, 255],  # 8: Blue
    [51, 153, 255],   # 9: Bright Blue
    [255, 153, 153],  # 10: Light Red
    [255, 102, 102],  # 11: Red
    [255, 51, 51],    # 12: Bright Red
    [153, 255, 153],  # 13: Light Green
    [102, 255, 102],  # 14: Green
    [51, 255, 51],    # 15: Bright Green
    [0, 255, 0],      # 16: Pure Green
    [0, 0, 255],      # 17: Blue
    [255, 0, 0],      # 18: Red
    [255, 255, 255],  # 19: White
]

# Ultralytics keypoint colors (17 keypoints)
KPT_COLOR = [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]

# Ultralytics limb colors (19 limbs)  
LIMB_COLOR = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]

def test_ultralytics_with_pt():
    """Test with ultralytics Python API using PT model for comparison"""
    print("=== Testing with Ultralytics Python API (PT Model) ===")
    
    # Load PT pose model through ultralytics
    model = YOLO(PT_MODEL_PATH)
    model.to(DEVICE)

    # Measure total inference time (includes preprocessing + inference + postprocessing)
    start_time = time.time()
    results = model(TEST_IMAGE_PATH, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=DEVICE)
    total_time = time.time() - start_time
    
    # Process results
    print(f"Ultralytics PT total time: {total_time:.3f}s")
    print("Note: Ultralytics automatically handles preprocessing and postprocessing internally")
    
    for r in results:
        print(f"Image: {r.path}")
        print(f"Image shape: {r.orig_img.shape}")
        
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            boxes = r.boxes
            keypoints = r.keypoints
            
            if boxes is not None and keypoints is not None:
                print(f"Number of pose detections: {len(boxes)}")
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    kpts = keypoints.xy[i].cpu().numpy()  # [17, 2]
                    kpts_conf = keypoints.conf[i].cpu().numpy()  # [17]
                    
                    x1, y1, x2, y2 = box.astype(int)
                    visible_keypoints = sum(1 for kc in kpts_conf if kc > 0.5)
                    print(f"  {i+1}. person: {conf:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2}) - Visible keypoints: {visible_keypoints}/17")
        else:
            print("No pose detections found")

def test_ultralytics_with_onnx():
    """Test with ultralytics Python API using ONNX model"""
    print("=== Testing with Ultralytics Python API (ONNX) ===")
    
    # Load ONNX pose model through ultralytics
    model = YOLO(ONNX_MODEL_PATH, task='pose')
    
    # Measure total inference time (includes preprocessing + inference + postprocessing)
    start_time = time.time()
    results = model(TEST_IMAGE_PATH, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    total_time = time.time() - start_time
    
    # Process results
    print(f"Ultralytics ONNX total time: {total_time:.3f}s")
    print("Note: Ultralytics automatically handles preprocessing and postprocessing internally")
    
    for r in results:
        print(f"Image: {r.path}")
        print(f"Image shape: {r.orig_img.shape}")
        
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            boxes = r.boxes
            keypoints = r.keypoints
            
            if boxes is not None and keypoints is not None:
                print(f"Number of pose detections: {len(boxes)}")
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    kpts = keypoints.xy[i].cpu().numpy()  # [17, 2]
                    kpts_conf = keypoints.conf[i].cpu().numpy()  # [17]
                    
                    x1, y1, x2, y2 = box.astype(int)
                    visible_keypoints = sum(1 for kc in kpts_conf if kc > 0.5)
                    print(f"  {i+1}. person: {conf:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2}) - Visible keypoints: {visible_keypoints}/17")
        else:
            print("No pose detections found")

def test_our_onnx_implementation():
    """Test our direct ONNX Runtime implementation with Ultralytics-compatible preprocessing/postprocessing"""
    print("\n=== Testing Our Direct ONNX Implementation (Ultralytics-Compatible) ===")
    
    # Load ONNX pose model directly with ONNXRuntime
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    
    # Load image
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        print("Error: Could not load bus.jpg")
        return
        
    original_height, original_width = image.shape[:2]
    
    # Preprocessing - Use Ultralytics LetterBox equivalent
    start_preprocess = time.time()
    
    # Convert BGR to RGB (like Ultralytics)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ultralytics LetterBox equivalent preprocessing
    input_size = 640
    shape = image_rgb.shape[:2]  # current shape [height, width]
    new_shape = (input_size, input_size)
    
    # Scale ratio (new / old) - same as Ultralytics
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding - same as Ultralytics
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    # Center padding (Ultralytics default: center=True)
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    # Resize if needed (using INTER_LINEAR like Ultralytics)
    if shape[::-1] != new_unpad:  # resize
        image_rgb = cv2.resize(image_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding (using padding_value=114 like Ultralytics)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_rgb = cv2.copyMakeBorder(image_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Normalize and convert to NCHW format
    input_tensor = image_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
    
    preprocess_time = time.time() - start_preprocess
    
    # Run inference
    start_inference = time.time()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = time.time() - start_inference
    
    # Complete postprocessing using main.py functions (same as Ultralytics level processing)
    start_postprocess = time.time()

    # Calculate padding for postprocessing (compatible with main.py format)
    pad = (left, top)  # (pad_width, pad_height)
    
    # Use main.py postprocess_pose function for complete processing
    detections = postprocess_pose(outputs, original_width, original_height, ratio, pad)
    postprocess_time = time.time() - start_postprocess
    
    # Print timing results
    total_time = preprocess_time + inference_time + postprocess_time
    print(f"Preprocessing time (Ultralytics-compatible): {preprocess_time:.3f}s")
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Complete postprocessing time: {postprocess_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Output shapes: {[output.shape for output in outputs]}")
    print(f"Number of pose detections (after NMS): {len(detections)}")
    
    # Print pose detection results
    for i, det in enumerate(detections):
        box = det['box']
        score = det['score']
        keypoints = det['keypoints']
        
        x1, y1, x2, y2 = box
        visible_keypoints = sum(1 for kpt in keypoints if kpt[2] > 0.5)
        
        print(f"  {i+1}. person: {score:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2}) - Visible keypoints: {visible_keypoints}/17")

    print("Note: This uses Ultralytics-compatible preprocessing and complete pose postprocessing with NMS")

def test_main_with_onnx():
    """Test using main.py functions with ONNX model for exact comparison"""
    print("\n=== Testing with main.py Functions (ONNX) ===")
    
    # Load ONNX pose model using ONNXRuntime (same as main.py)
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    
    # Load image
    image_path = TEST_IMAGE_PATH
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load bus.jpg")
        return
        
    original_height, original_width = image.shape[:2]
    print(f"Original image shape: {image.shape}")
    
    # Preprocess using main.py letterbox function
    start_preprocess = time.time()
    processed_image, ratio, pad = letterbox(image, new_shape=(640, 640))
    
    # Convert to model input format
    input_tensor = processed_image.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
    preprocess_time = time.time() - start_preprocess
    
    # Run inference
    start_inference = time.time()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = time.time() - start_inference
    
    # Post-process using main.py pose functions
    start_postprocess = time.time()
    
    # Format outputs to match main.py expected format
    detections = postprocess_pose(outputs, original_width, original_height, ratio, pad)
    postprocess_time = time.time() - start_postprocess
    
    # Print timing results
    total_time = preprocess_time + inference_time + postprocess_time
    print(f"Preprocessing time: {preprocess_time:.3f}s")
    print(f"Inference time: {inference_time:.3f}s") 
    print(f"Postprocessing time: {postprocess_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    
    # Print pose detection results
    print(f"Number of pose detections: {len(detections)}")
    
    for i, det in enumerate(detections):
        box = det['box']
        score = det['score']
        keypoints = det['keypoints']
        
        x1, y1, x2, y2 = box
        visible_keypoints = sum(1 for kpt in keypoints if kpt[2] > 0.5)
        
        print(f"  {i+1}. person: {score:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2}) - Visible keypoints: {visible_keypoints}/17")

if __name__ == "__main__":
    print("🚀 Performance Comparison: PT vs ONNX Models (Pose Estimation)")
    print("=" * 80)
    
    # Test PT pose model
    test_ultralytics_with_pt()
    
    print("\n" + "=" * 80)
    
    # Test ONNX pose model through ultralytics
    test_ultralytics_with_onnx()
    
    print("\n" + "=" * 80)

    # Test our direct ONNX implementation (complete pipeline)
    test_our_onnx_implementation()
    
    print("\n" + "=" * 80)
    
    # Test main.py functions with ONNX
    test_main_with_onnx()

    print("\n" + "=" * 80)
    print("🏁 Pose estimation performance comparison complete!")
    print("Note: Times may vary between runs due to system load and thermal throttling.")