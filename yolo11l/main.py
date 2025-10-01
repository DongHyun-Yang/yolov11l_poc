import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
from pathlib import Path

# ONNX model and image file path configuration
ONNX_MODEL_PATH = 'models/yolo11l.onnx'
IMAGE_PATH = '../assets/' # Change to desired image file or directory for detection.
# IMAGE_PATH = '../test_images/'
OUTPUT_DIR = 'output'  # Directory to save results
CONFIDENCE_THRESHOLD = 0.25  # Standard threshold matching ultralytics default
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.25   # Standard threshold matching ultralytics default

# COCO class names (based on dataset that YOLOv11 was trained on)
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """Resize image to square while maintaining aspect ratio."""
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
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

def preprocess_image(image_path):
    """Read image and preprocess it for model input."""
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # Accurate preprocessing using letterbox
    image, ratio, (dw, dh) = letterbox(image, new_shape=(640, 640))

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

def postprocess_output(outputs, original_width, original_height, ratio, pad):
    """Process model output to get final bounding boxes."""
    outputs = np.squeeze(outputs).T
    dw, dh = pad

    boxes = []
    scores = []
    class_ids = []

    # Process each detection result
    for row in outputs:
        # Confidence scores for each class
        class_scores = row[4:]
        
        # Find class ID with highest score
        class_id = np.argmax(class_scores)
        max_score = class_scores[class_id]
        
        if max_score > CONFIDENCE_THRESHOLD:
            # Bounding box coordinates (cx, cy, w, h)
            cx, cy, w, h = row[:4]
            
            # Remove padding and convert to original size
            cx = (cx - dw) / ratio[0]
            cy = (cy - dh) / ratio[1]
            w = w / ratio[0]
            h = h / ratio[1]
            
            # Convert center coordinates to corner coordinates
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            # Limit within image boundaries
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(max_score)
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, np.array(scores), SCORE_THRESHOLD, IOU_THRESHOLD)
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append({
                    'box': boxes[i],
                    'score': scores[i],
                    'class_id': class_ids[i]
                })
                
        return final_boxes
    else:
        return []

def draw_detections(image_path, detections, output_path):
    """Draw detected objects on image."""
    image = cv2.imread(image_path)
    
    # Color palette (different colors for each class)
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    for det in detections:
        box = det['box']
        score = det['score']
        class_id = det['class_id']
        
        x1, y1, x2, y2 = box
        label = f"{CLASSES[class_id]}: {score:.2f}"
        
        # Use class-specific color
        color = colors[class_id].tolist()
        
        # Draw bounding box (thicker line)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Label background rectangle
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Label text (in white)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save result image
    cv2.imwrite(output_path, image)
    print(f"Detection result saved to '{output_path}' file.")

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
        detections = postprocess_output(outputs, original_w, original_h, ratio, pad)
        
        # 4. Generate result save path
        filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{filename}_detected.jpg")
        
        # 5. Visualization
        if detections:
            draw_detections(image_path, detections, output_path)
            print(f"[{filename}] Total {len(detections)} objects detected.")
            for i, det in enumerate(detections):
                class_name = CLASSES[det['class_id']]
                score = det['score']
                box = det['box']
                print(f"  {i+1}. {class_name}: {score:.2f} - Position: ({box[0]}, {box[1]}) ~ ({box[2]}, {box[3]})")
        else:
            # Copy and save original image even if no objects detected
            image = cv2.imread(image_path)
            cv2.imwrite(output_path, image)
            print(f"[{filename}] No objects detected.")
            
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
    
    print(f"Processing {len(image_files)} image files in total.")
    print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
    print("-" * 50)
    
    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {Path(image_path).name}")
        if process_single_image(image_path, OUTPUT_DIR):
            success_count += 1
    
    print("-" * 50)
    print(f"Processing completed: {success_count}/{len(image_files)} successful")
    print(f"Result files saved in '{OUTPUT_DIR}' folder.")