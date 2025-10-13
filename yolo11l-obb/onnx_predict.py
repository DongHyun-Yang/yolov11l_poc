#!/usr/bin/env python3
"""
Pure ONNX Runtime OBB Detection Script
Ultralytics YOLO OBB 모델을 ONNX Runtime으로 직접 실행하여 예측하는 스크립트
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
import time
import json


# DOTAv1.0 class names (YOLOv11-obb 학습 데이터셋)
CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 
    'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 
    'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
]

# 설정값
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 1024
MAX_DETECTIONS = 300


def letterbox(img, new_shape=(1024, 1024), color=(114, 114, 114), auto=True, 
              scaleFill=False, scaleup=True, stride=32):
    """
    Ultralytics letterbox 전처리 함수 복제
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)


def preprocess_image(image_path, input_size=1024):
    """
    이미지 전처리: Ultralytics 스타일
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_shape = img.shape[:2]  # H, W
    
    # Letterbox 전처리
    img_processed, ratio, pad = letterbox(img, new_shape=(input_size, input_size), auto=False, stride=32)
    
    # BGR to RGB 변환
    img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    
    # 정규화 (0-255 -> 0-1)
    img_processed = img_processed.astype(np.float32) / 255.0
    
    # HWC -> CHW 변환
    img_processed = np.transpose(img_processed, (2, 0, 1))
    
    # 배치 차원 추가
    img_processed = np.expand_dims(img_processed, axis=0)
    
    return img_processed, original_shape, ratio, pad


def xywhr2xyxyxyxy(rboxes):
    """
    회전된 박스 (x, y, w, h, r)를 4개 모서리 좌표로 변환
    """
    cos_r = np.cos(rboxes[..., 4])
    sin_r = np.sin(rboxes[..., 4])
    
    # Get half width and height
    h_w = rboxes[..., 2] / 2
    h_h = rboxes[..., 3] / 2
    
    # Calculate corner offsets
    dx1 = -h_w * cos_r - (-h_h) * sin_r  # Top-left
    dy1 = -h_w * sin_r + (-h_h) * cos_r
    
    dx2 = h_w * cos_r - (-h_h) * sin_r   # Top-right
    dy2 = h_w * sin_r + (-h_h) * cos_r
    
    dx3 = h_w * cos_r - h_h * sin_r      # Bottom-right  
    dy3 = h_w * sin_r + h_h * cos_r
    
    dx4 = -h_w * cos_r - h_h * sin_r     # Bottom-left
    dy4 = -h_w * sin_r + h_h * cos_r
    
    # Add center coordinates
    x1 = rboxes[..., 0] + dx1
    y1 = rboxes[..., 1] + dy1
    x2 = rboxes[..., 0] + dx2
    y2 = rboxes[..., 1] + dy2
    x3 = rboxes[..., 0] + dx3
    y3 = rboxes[..., 1] + dy3
    x4 = rboxes[..., 0] + dx4
    y4 = rboxes[..., 1] + dy4
    
    return np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)


def regularize_rboxes(rboxes):
    """
    회전된 박스 각도 정규화: [-π/4, 3π/4] 범위로 제한
    """
    x, y, w, h, r = rboxes[..., 0], rboxes[..., 1], rboxes[..., 2], rboxes[..., 3], rboxes[..., 4]
    
    # Normalize angle to [-π, π] range
    r = r % (2 * np.pi)
    r = np.where(r > np.pi, r - 2 * np.pi, r)
    
    # Regularize to [-π/4, 3π/4] range
    # If angle > 3π/4 or angle < -π/4, rotate by π/2 and swap w,h
    mask = (r > 3 * np.pi / 4) | (r < -np.pi / 4)
    
    # Swap width and height for regularized boxes
    w_new = np.where(mask, h, w)
    h_new = np.where(mask, w, h) 
    
    # Adjust angle
    r_new = np.where(mask, r - np.pi / 2, r)
    r_new = np.where(r_new < -np.pi / 4, r_new + np.pi / 2, r_new)
    
    return np.stack([x, y, w_new, h_new, r_new], axis=-1)


def non_max_suppression_obb(predictions, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    OBB Non-Maximum Suppression
    """
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Remove batch dimension
    
    # Transpose if needed: (20, 21504) -> (21504, 20)
    if predictions.shape[0] == 20:
        predictions = predictions.T
    
    # Extract components
    boxes = predictions[:, :4]  # x, y, w, h
    angles = predictions[:, 4:5]  # angle
    scores = predictions[:, 5:]  # class scores (15 classes)
    
    # Apply sigmoid to class scores (ONNX outputs raw logits)
    scores = 1 / (1 + np.exp(-scores))
    
    # Get confidence (max class score) and class predictions
    conf = np.max(scores, axis=1)
    class_preds = np.argmax(scores, axis=1)
    
    # Filter by confidence threshold
    valid_mask = conf >= conf_thres
    if not np.any(valid_mask):
        return np.array([]).reshape(0, 7)
    
    boxes = boxes[valid_mask]
    angles = angles[valid_mask]
    conf = conf[valid_mask]
    class_preds = class_preds[valid_mask]
    
    # Sort by confidence (descending)
    sort_idx = np.argsort(-conf)
    boxes = boxes[sort_idx]
    angles = angles[sort_idx]
    conf = conf[sort_idx]
    class_preds = class_preds[sort_idx]
    
    # Convert to xyxy for NMS (approximation)
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Apply NMS using OpenCV
    obb_iou_thres = min(iou_thres, 0.3)  # Lower threshold for OBB
    indices = cv2.dnn.NMSBoxes(
        xyxy_boxes.tolist(), 
        conf.tolist(), 
        conf_thres, 
        obb_iou_thres
    )
    
    if isinstance(indices, np.ndarray) and len(indices) > 0:
        indices = indices.flatten()
        
        # Limit detections
        if len(indices) > max_det:
            indices = indices[:max_det]
        
        # Combine results: [x, y, w, h, angle, conf, cls]
        results = np.zeros((len(indices), 7))
        for i, idx in enumerate(indices):
            results[i, :4] = boxes[idx]      # x, y, w, h
            results[i, 4] = angles[idx, 0]   # angle
            results[i, 5] = conf[idx]        # confidence
            results[i, 6] = class_preds[idx] # class
        
        return results
    else:
        return np.array([]).reshape(0, 7)


def scale_boxes(img_shape, boxes, img0_shape, ratio_pad=None):
    """
    박스 좌표를 원본 이미지 크기로 스케일링
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img_shape[0] / img0_shape[0], img_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img_shape[1] - img0_shape[1] * gain) / 2, (img_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] /= gain
    boxes[..., [0, 2]] /= gain
    boxes[..., [1, 3]] -= pad[1]  # y padding
    
    return boxes


def draw_obb_detections(image_path, detections, output_path):
    """
    OBB 탐지 결과를 이미지에 그리기
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    for det in detections:
        x, y, w, h, angle, conf, cls_id = det
        class_name = CLASSES[int(cls_id)]
        
        # Create rotated rectangle
        center = (int(x), int(y))
        size = (int(w), int(h))
        angle_deg = np.degrees(angle)
        
        # Get 4 corner points of rotated rectangle
        rect = cv2.boxPoints(((x, y), (w, h), angle_deg))
        rect = np.int0(rect)
        
        # Draw OBB
        cv2.drawContours(img, [rect], 0, (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (int(x), int(y-25)), (int(x + label_size[0]), int(y)), (0, 255, 0), -1)
        cv2.putText(img, label, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


def run_onnx_prediction(model_path, image_path, output_dir="runs/obb/predict_onnx"):
    """
    ONNX Runtime을 사용한 OBB 예측 메인 함수
    """
    print("=" * 80)
    print("YOLO OBB ONNX Runtime Prediction")
    print("=" * 80)
    
    # ONNX 모델 로드
    print(f"Loading ONNX model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    
    # ONNX Runtime 세션 생성
    providers = ['CPUExecutionProvider']
    if ort.get_device() == 'GPU':
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 모델 입력/출력 정보
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Model input shape: {input_shape}")
    
    # 이미지 전처리
    print(f"Preprocessing image: {image_path}")
    start_time = time.time()
    input_tensor, original_shape, ratio, pad = preprocess_image(image_path, INPUT_SIZE)
    preprocess_time = time.time() - start_time
    
    # 추론 실행
    print(f"Running inference...")
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    inference_time = time.time() - start_time
    
    # 후처리
    print(f"Post-processing results...")
    start_time = time.time()
    detections = non_max_suppression_obb(
        outputs, 
        conf_thres=CONFIDENCE_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS
    )
    
    if len(detections) > 0:
        # 박스 좌표를 원본 이미지 크기로 변환
        rboxes = detections[:, :5].copy()  # x, y, w, h, angle
        rboxes[:, :4] = scale_boxes((INPUT_SIZE, INPUT_SIZE), rboxes[:, :4], original_shape, (ratio, pad))
        
        # 정규화된 박스 생성
        rboxes = regularize_rboxes(rboxes)
        detections[:, :5] = rboxes
    
    postprocess_time = time.time() - start_time
    
    # 결과 출력
    print(f"\nResults:")
    print(f"Raw output shape: {outputs.shape}")
    print(f"Total OBB detections: {len(detections)}")
    
    if len(detections) > 0:
        confidences = detections[:, 5]
        classes = detections[:, 6].astype(int)
        
        print(f"Confidence range: {confidences.min():.3f} ~ {confidences.max():.3f}")
        print(f"Confidences >= {CONFIDENCE_THRESHOLD}: {(confidences >= CONFIDENCE_THRESHOLD).sum()}")
        
        # 클래스 분포
        unique_classes, counts = np.unique(classes, return_counts=True)
        class_dist = dict(zip(unique_classes, counts))
        print(f"Class distribution: {class_dist}")
        
        # 성능 메트릭
        print(f"\nPerformance:")
        print(f"Preprocess: {preprocess_time*1000:.1f}ms")
        print(f"Inference: {inference_time*1000:.1f}ms") 
        print(f"Postprocess: {postprocess_time*1000:.1f}ms")
        print(f"Total: {(preprocess_time + inference_time + postprocess_time)*1000:.1f}ms")
        
        # 결과 이미지 저장
        os.makedirs(output_dir, exist_ok=True)
        filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{filename}_obb_detected.jpg")
        
        draw_obb_detections(image_path, detections, output_path)
        print(f"\nResult saved to: {output_path}")
        
        # 탐지 결과 상세 출력
        print(f"\nDetections:")
        for i, det in enumerate(detections[:10]):  # 상위 10개만 출력
            x, y, w, h, angle, conf, cls_id = det
            class_name = CLASSES[int(cls_id)]
            print(f"  {i+1}. {class_name}: {conf:.3f} - Center: ({x:.1f}, {y:.1f}) - Size: {w:.1f}x{h:.1f} - Angle: {np.degrees(angle):.1f}°")
    
    else:
        print("No detections found")
    
    return detections


if __name__ == "__main__":
    # 설정
    MODEL_PATH = "yolo11l-obb/models/yolo11l-obb.onnx"
    IMAGE_PATH = "assets/boats.jpg"
    OUTPUT_DIR = "runs/obb/predict_onnx"
    
    try:
        # 예측 실행
        detections = run_onnx_prediction(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR)
        
        print(f"\n✅ Prediction completed successfully!")
        print(f"Found {len(detections)} objects")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()