import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
from pathlib import Path

# ONNX model and image file path configuration
ONNX_MODEL_PATH = 'models/yolo11l-obb.onnx'
IMAGE_PATH = '../assets/boats.jpg' # Change to desired image file or directory for detection.
# IMAGE_PATH = '../test_images/'
OUTPUT_DIR = 'output'  # Directory to save results
CONFIDENCE_THRESHOLD = 0.25  # Ultralytics default conf threshold
IOU_THRESHOLD = 0.45         # Ultralytics default IoU threshold  
SCORE_THRESHOLD = 0.25       # Same as confidence threshold

# DOTAv1.0 class names (dataset that YOLOv11-obb was trained on)
CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 
           'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 
           'roundabout', 'soccer-ball-field', 'swimming-pool']

def letterbox(image, new_shape=(1024, 1024), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints.
    Exact replica of Ultralytics letterbox function.
    """
    shape = image.shape[:2]  # current shape [height, width]
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
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return image, ratio, (dw, dh)

def preprocess_image(image_path, imgsz=1024):
    """Read image and preprocess it for model input exactly like Ultralytics."""
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # ONNX model requires exact 1024x1024 input, so use auto=False
    image, ratio, (dw, dh) = letterbox(image, new_shape=(imgsz, imgsz), auto=False, stride=32)

    # BGR -> RGB, HWC -> CHW, normalization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32) / 255.0
    
    return image, original_width, original_height, ratio, (dw, dh)

def run_inference(model_path, input_tensor):
    """Run inference with ONNX model."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    outputs = session.run([output_name], {input_name: input_tensor})
    return outputs[0]

def regularize_rboxes(rboxes):
    """
    Regularize rotated bounding boxes to range [0, pi/2].
    Exact implementation from Ultralytics ops.py
    
    Args:
        rboxes: List or array [x, y, w, h, angle] format
    Returns:
        Regularized rbox [x, y, w, h, angle]
    """
    if isinstance(rboxes, list):
        x, y, w, h, t = rboxes
    else:
        x, y, w, h, t = rboxes[0], rboxes[1], rboxes[2], rboxes[3], rboxes[4]
    
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = (t % np.pi) >= (np.pi / 2)
    if swap:
        w, h = h, w  # swap width and height
    
    t = t % (np.pi / 2)  # normalize angle to [0, π/2]
    
    return [x, y, w, h, t]

# Remove complex rotated IoU - use standard NMS like Ultralytics with rotated=False


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, xywh=False):
    """
    Scale boxes from img1_shape to img0_shape.
    Based on Ultralytics ops.scale_boxes function.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if xywh:
        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
    else:
        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
    
    return boxes


def non_max_suppression_obb(predictions, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.
    Mimics Ultralytics non_max_suppression for OBB.
    
    Args:
        predictions: (tensor) predictions from model, shape = [batch_size, num_anchors, 5 + num_classes + 1]
        conf_thres: (float) confidence threshold
        iou_thres: (float) IoU threshold for NMS
        classes: (list[int]) filter by class
        agnostic: (bool) when True, the model is agnostic to the number of classes
        multi_label: (bool) when True, each box may have multiple labels
        max_det: (int) maximum number of detections to keep

    Returns:
        list of detections, each item a tensor of shape (num_boxes, 7) where 7 = [x, y, w, h, angle, conf, cls]
    """
    predictions = np.array(predictions)
    if len(predictions.shape) == 3:
        predictions = np.squeeze(predictions, 0)
    
    # Transpose to [num_detections, 20] if needed
    if predictions.shape[0] == 20:
        predictions = predictions.T
    
    # Get number of classes
    nc = predictions.shape[1] - 5  # number of classes = total - [x, y, w, h, angle]
    
    # Extract boxes, scores, angles
    boxes = predictions[:, :4]  # x, y, w, h
    angles = predictions[:, 4:5]  # angle
    scores = predictions[:, 5:]  # class scores
    
    # Apply sigmoid to class scores to get probabilities (ONNX models often output raw logits)
    scores = 1 / (1 + np.exp(-scores))  # sigmoid activation
    
    # Compute conf = max(class_scores)
    conf = np.max(scores, axis=1)
    
    # Filter by confidence threshold
    conf_mask = conf > conf_thres
    
    if not np.any(conf_mask):
        return []
    
    # Apply confidence mask
    boxes = boxes[conf_mask]
    angles = angles[conf_mask] 
    scores = scores[conf_mask]
    conf = conf[conf_mask]
    
    # Get class predictions
    class_preds = np.argmax(scores, axis=1)
    
    # Debug: Print class distribution before NMS
    unique_classes_before, counts_before = np.unique(class_preds, return_counts=True)
    print(f"Classes before NMS: {dict(zip(unique_classes_before.astype(int), counts_before))}")
    print(f"Sample class scores (first 5): {scores[:5]}")
    print(f"Sample class predictions (first 5): {class_preds[:5]}")
    
    # Convert boxes from center format to corner format for NMS
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # For OBB, Ultralytics typically uses lower IoU threshold when rotated=True
    obb_iou_thres = min(iou_thres, 0.3)  # Lower threshold for better OBB handling
    
    # Sort by confidence before NMS (highest confidence first)
    conf_order = np.argsort(-conf)
    boxes = boxes[conf_order]
    angles = angles[conf_order]
    conf = conf[conf_order]
    class_preds = class_preds[conf_order]
    xyxy_boxes = xyxy_boxes[conf_order]
    
    # Apply NMS with OBB-optimized threshold
    indices = cv2.dnn.NMSBoxes(xyxy_boxes.tolist(), conf.tolist(), conf_thres, obb_iou_thres)
    
    if isinstance(indices, np.ndarray) and len(indices) > 0:
        indices = indices.flatten()
        
        # Limit detections to match Ultralytics default (typically fewer for OBB)
        ultralytics_max_det = min(max_det, 300)  # Ensure we don't exceed typical limits
        if len(indices) > ultralytics_max_det:
            indices = indices[:ultralytics_max_det]
        
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

def postprocess_obb(outputs, original_width, original_height, ratio, pad):
    """
    Process OBB model output exactly like Ultralytics.
    """
    print(f"Raw output shape: {outputs.shape}")
    
    # Apply NMS like Ultralytics with OBB-specific parameters
    pred = non_max_suppression_obb(
        outputs, 
        conf_thres=CONFIDENCE_THRESHOLD, 
        iou_thres=IOU_THRESHOLD, 
        max_det=300  # Ultralytics default for OBB
    )
    
    if len(pred) == 0:
        print("No detections found")
        return []
    
    print(f"Total OBB detections: {len(pred)}")
    print(f"OBB tensor shape: {pred.shape}")
    
    # Get confidence values and print analysis like test.py
    confidences = pred[:, 5]
    print(f"Confidence range: {np.min(confidences):.3f} ~ {np.max(confidences):.3f}")
    print(f"Confidences >= 0.25: {np.sum(confidences >= 0.25)}")
    
    # Check class distribution
    classes = pred[:, 6].astype(int)
    unique_classes, counts = np.unique(classes, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, counts))}")
    
    # More detailed conf analysis like test.py
    conf_bins = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(conf_bins)-1):
        count = np.sum((confidences >= conf_bins[i]) & (confidences < conf_bins[i+1]))
        print(f"Conf {conf_bins[i]:.1f}~{conf_bins[i+1]:.1f}: {count}")
    print(f"Conf >= 0.9: {np.sum(confidences >= 0.9)}")
    
    dw, dh = pad
    
    # Process detections
    final_boxes = []
    for detection in pred:
        cx, cy, w, h, angle, conf, class_id = detection
        
        # Remove letterbox padding and scale to original image coordinates
        cx = (cx - dw) / ratio[0]
        cy = (cy - dh) / ratio[1] 
        w = w / ratio[0]
        h = h / ratio[1]
        
        # Apply regularize_rboxes like Ultralytics
        regularized_rbox = regularize_rboxes([cx, cy, w, h, angle])
        
        # Convert to corner coordinates for visualization
        x1 = max(0, min(cx - w/2, original_width))
        y1 = max(0, min(cy - h/2, original_height))
        x2 = max(0, min(cx + w/2, original_width))
        y2 = max(0, min(cy + h/2, original_height))
        
        final_boxes.append({
            'box': [x1, y1, x2, y2],
            'obb': regularized_rbox,
            'score': conf,
            'class_id': int(class_id)
        })
    
    return final_boxes

def draw_obb_detections(image_path, detections, output_path):
    """Draw oriented bounding boxes on image."""
    image = cv2.imread(image_path)
    
    # Color palette (different colors for each class)
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    for det in detections:
        box = det['box']  # Corner coordinates for fallback
        obb = det['obb']  # OBB parameters [cx, cy, w, h, angle]
        score = det['score']
        class_id = det['class_id']
        
        label = f"{CLASSES[class_id]}: {score:.2f}"
        color = colors[class_id].tolist()
        
        # Extract OBB parameters
        cx, cy, w, h, angle = obb
        
        # Create rotated rectangle
        center = (int(cx), int(cy))
        size = (int(w), int(h))
        
        # Get the 4 corner points of the rotated rectangle
        # Convert angle from radians to degrees for OpenCV
        rect = ((cx, cy), (w, h), np.degrees(angle))
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)
        
        # Draw oriented bounding box
        cv2.drawContours(image, [box_points], 0, color, 3)
        
        # Draw center point
        cv2.circle(image, center, 3, color, -1)
        
        # Label position (use top-left corner of regular bounding box)
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Ensure integer coordinates
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save result image
    cv2.imwrite(output_path, image)
    print(f"OBB detection result saved to '{output_path}' file.")

def get_image_files(path):
    """Find image files in the path."""
    image_extensions = ['*.jpg', '*.jpeg']
    image_files = []
    
    if os.path.isfile(path):
        # Single file case
        if any(path.lower().endswith(ext[1:]) for ext in image_extensions):
            image_files.append(path)
    elif os.path.isdir(path):
        # Directory case - use set to avoid duplicates
        image_files_set = set()
        for ext in image_extensions:
            # Search for both lowercase and uppercase versions
            image_files_set.update(glob.glob(os.path.join(path, ext)))
            image_files_set.update(glob.glob(os.path.join(path, ext.upper())))
        image_files = list(image_files_set)
    
    return image_files

def process_single_image(image_path, output_dir):
    """Process a single image."""
    try:
        # 1. Preprocessing
        input_tensor, original_w, original_h, ratio, pad = preprocess_image(image_path)
        
        # 2. Inference
        outputs = run_inference(ONNX_MODEL_PATH, input_tensor)
        
        # 3. Post-processing
        detections = postprocess_obb(outputs, original_w, original_h, ratio, pad)
        
        # 4. Generate result save path
        filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{filename}_obb_detected.jpg")
        
        # 5. Visualization
        if detections:
            draw_obb_detections(image_path, detections, output_path)
            print(f"[{filename}] Total {len(detections)} oriented objects detected.")
            for i, det in enumerate(detections):
                class_name = CLASSES[det['class_id']]
                score = det['score']
                box = det['box']
                obb = det['obb']
                print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({box[0]}, {box[1]}) ~ ({box[2]}, {box[3]}) - OBB: ({obb[0]:.1f}, {obb[1]:.1f}, {obb[2]:.1f}x{obb[3]:.1f}, {obb[4]:.1f}°)")
        else:
            # Copy and save original image even if no objects detected
            image = cv2.imread(image_path)
            cv2.imwrite(output_path, image)
            print(f"[{filename}] No oriented objects detected.")
            
        return True
        
    except Exception as e:
        print(f"[{Path(image_path).name}] Error occurred during processing: {str(e)}")
        return False

if __name__ == '__main__':
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find image files
    image_files = get_image_files(IMAGE_PATH)
    
    if not image_files:
        print(f"No image files found in '{IMAGE_PATH}'.")
        print("Supported formats: .jpg, .jpeg")
        exit(1)
    
    print(f"Processing {len(image_files)} image files for OBB detection.")
    print(f"Input size: 1024x1024 (optimized for OBB detection)")
    print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
    print("-" * 50)
    
    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {Path(image_path).name}")
        if process_single_image(image_path, OUTPUT_DIR):
            success_count += 1
    
    print("-" * 50)
    print(f"OBB detection completed: {success_count}/{len(image_files)} successful")
    print(f"Result files saved in '{OUTPUT_DIR}' folder.")
