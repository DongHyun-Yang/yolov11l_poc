#!/usr/bin/env python3
"""
Compare detection results between ultralytics and our custom implementation
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
from main import letterbox, postprocess_output

PT_MODEL_PATH = 'models/yolo11l.pt'
ONNX_MODEL_PATH = 'models/yolo11l.onnx'
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
DEVICE = 'cpu'  # Force CPU for consistency
TEST_IMAGE_PATH = '../assets/bus.jpg'

def test_ultralytics_with_pt():
    """Test with ultralytics Python API using PT model for comparison"""
    print("=== Testing with Ultralytics Python API (PT Model) ===")
    
    # Load PT model through ultralytics
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
        
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            
            print(f"Number of detections: {len(boxes)}")
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = model.names[cls_id]
                x1, y1, x2, y2 = box.astype(int)
                print(f"  {i+1}. {class_name}: {conf:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2})")

def test_ultralytics_with_onnx():
    """Test with ultralytics Python API using ONNX model"""
    print("=== Testing with Ultralytics Python API (ONNX) ===")
    
    # Load ONNX model through ultralytics
    model = YOLO(ONNX_MODEL_PATH)
    
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
        
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            
            print(f"Number of detections: {len(boxes)}")
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = model.names[cls_id]
                x1, y1, x2, y2 = box.astype(int)
                print(f"  {i+1}. {class_name}: {conf:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2})")

def test_our_onnx_implementation():
    """Test our direct ONNX Runtime implementation with Ultralytics-compatible preprocessing/postprocessing"""
    print("\n=== Testing Our Direct ONNX Implementation (Ultralytics-Compatible) ===")
    
    # Load ONNX model directly with ONNXRuntime
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
    
    # Use main.py postprocess_output function for complete processing
    detections = postprocess_output(outputs, original_width, original_height, ratio, pad)
    postprocess_time = time.time() - start_postprocess
    
    # Print timing results
    total_time = preprocess_time + inference_time + postprocess_time
    print(f"Preprocessing time (Ultralytics-compatible): {preprocess_time:.3f}s")
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Complete postprocessing time: {postprocess_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Output shapes: {[output.shape for output in outputs]}")
    print(f"Number of detections (after NMS): {len(detections)}")
    
    # Print detection results like Ultralytics
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    for i, det in enumerate(detections):
        box = det['box']
        score = det['score']
        class_id = det['class_id']
        
        x1, y1, x2, y2 = box
        class_name = CLASSES[class_id]
        
        print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2})")

    print("Note: This uses Ultralytics-compatible preprocessing and complete postprocessing with NMS")

def test_main_with_onnx():
    """Test using main.py functions with ONNX model for exact comparison"""
    print("\n=== Testing with main.py Functions (ONNX) ===")
    
    # Load ONNX model using ONNXRuntime (same as main.py)
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
    
    # Post-process using main.py functions
    start_postprocess = time.time()
    
    # Format outputs to match main.py expected format
    detections = postprocess_output(outputs, original_width, original_height, ratio, pad)
    postprocess_time = time.time() - start_postprocess
    
    # Print timing results
    total_time = preprocess_time + inference_time + postprocess_time
    print(f"Preprocessing time: {preprocess_time:.3f}s")
    print(f"Inference time: {inference_time:.3f}s") 
    print(f"Postprocessing time: {postprocess_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    
    # Print detection results
    print(f"Number of detections: {len(detections)}")
    
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    for i, det in enumerate(detections):
        box = det['box']
        score = det['score']
        class_id = det['class_id']
        
        x1, y1, x2, y2 = box
        class_name = CLASSES[class_id]
        
        print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({x1}, {y1}) ~ ({x2}, {y2})")

if __name__ == "__main__":
    print("🚀 Performance Comparison: PT vs ONNX Models (Object Detection)")
    print("=" * 80)
    
    # Test PT model
    test_ultralytics_with_pt()
    
    print("\n" + "=" * 80)
    
    # Test ONNX model through ultralytics
    test_ultralytics_with_onnx()
    
    print("\n" + "=" * 80)

    # Test our direct ONNX implementation (complete pipeline)
    test_our_onnx_implementation()
    
    print("\n" + "=" * 80)
    
    # Test main.py functions with ONNX
    test_main_with_onnx()

    print("\n" + "=" * 80)
    print("🏁 Performance comparison complete!")
    print("Note: Times may vary between runs due to system load and thermal throttling.")