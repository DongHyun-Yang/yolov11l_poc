import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
from pathlib import Path

# ONNX model and image file path configuration
ONNX_MODEL_PATH = 'models/yolo11l-pose.onnx'
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

# COCO Pose keypoints (17 keypoints for human pose)
POSE_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
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
    outputs = session.run(None, {input_name: input_tensor})
    return outputs

def postprocess_pose(outputs, original_width, original_height, ratio, pad):
    """Process pose estimation model output to get final poses."""
    # YOLOv11-pose output: [batch, 56, 8400] (4 box + 1 conf + 51 keypoints)
    # keypoints: 17 points * 3 (x, y, visibility) = 51
    
    if len(outputs) == 0:
        return []
    
    predictions = outputs[0]  # Shape: [1, 56, 8400]
    predictions = np.squeeze(predictions).T  # Shape: [8400, 56]
    
    dw, dh = pad
    boxes = []
    scores = []
    keypoints_list = []

    # Process each detection result
    for row in predictions:
        # Split the prediction: [x, y, w, h, conf, keypoints...]
        box_data = row[:4]  # x, y, w, h
        conf = row[4]       # confidence
        keypoints_data = row[5:]  # 51 keypoint values (17 * 3)
        
        if conf > CONFIDENCE_THRESHOLD:
            # Bounding box coordinates (cx, cy, w, h)
            cx, cy, w, h = box_data
            
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
            
            # Process keypoints (17 keypoints * 3 values each)
            keypoints = []
            for i in range(17):
                kpt_x = keypoints_data[i * 3]     # x coordinate
                kpt_y = keypoints_data[i * 3 + 1] # y coordinate
                kpt_conf = keypoints_data[i * 3 + 2] # confidence/visibility
                
                # Remove padding and convert to original size
                kpt_x = (kpt_x - dw) / ratio[0]
                kpt_y = (kpt_y - dh) / ratio[1]
                
                # Limit within image boundaries
                kpt_x = max(0, min(kpt_x, original_width))
                kpt_y = max(0, min(kpt_y, original_height))
                
                keypoints.append([kpt_x, kpt_y, kpt_conf])
            
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            keypoints_list.append(keypoints)

    # Apply Non-Maximum Suppression
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, np.array(scores), SCORE_THRESHOLD, IOU_THRESHOLD)
        
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append({
                    'box': boxes[i],
                    'score': scores[i],
                    'class_id': 0,  # person class for pose
                    'keypoints': keypoints_list[i]
                })
                
        return final_detections
    else:
        return []

def draw_pose(image_path, detections, output_path):
    """Draw pose estimation results on image (matching Ultralytics style exactly)."""
    image = cv2.imread(image_path)
    
    # Line width calculation (matching ultralytics)
    line_width = max(round(sum(image.shape) / 2 * 0.003), 2)
    
    for det in detections:
        box = det['box']
        score = det['score']
        keypoints = det['keypoints']
        
        x1, y1, x2, y2 = box
        label = f"person {score:.2f}"
        
        # Draw bounding box (matching ultralytics colors)
        box_color = (255, 144, 30)  # Ultralytics default box color (BGR)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, line_width)
        
        # Label background and text
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, max(line_width - 1, 1))[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), box_color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), max(line_width - 1, 1))
        
        # Convert keypoints format for ultralytics-style drawing
        kpts = np.array(keypoints)  # Shape: [17, 3]
        
        # Draw keypoints (matching ultralytics colors and style)
        radius = max(line_width - 1, 1)
        conf_thres = 0.25
        
        for i, (kx, ky, kconf) in enumerate(kpts):
            if kconf > conf_thres:
                # Get keypoint color from ultralytics palette
                color_idx = KPT_COLOR[i]
                color = POSE_PALETTE[color_idx]
                # Convert RGB to BGR for OpenCV
                color_bgr = (color[2], color[1], color[0])
                cv2.circle(image, (int(kx), int(ky)), radius, color_bgr, -1, lineType=cv2.LINE_AA)
        
        # Draw skeleton connections (matching ultralytics style)
        for i, (start_idx, end_idx) in enumerate(POSE_SKELETON):
            # Adjust indices (ultralytics uses 1-based, we use 0-based)
            start_idx_adj = start_idx - 1
            end_idx_adj = end_idx - 1
            
            if 0 <= start_idx_adj < len(kpts) and 0 <= end_idx_adj < len(kpts):
                start_kpt = kpts[start_idx_adj]
                end_kpt = kpts[end_idx_adj]
                
                # Check confidence and bounds
                if (start_kpt[2] > conf_thres and end_kpt[2] > conf_thres and
                    start_kpt[0] > 0 and start_kpt[1] > 0 and
                    end_kpt[0] > 0 and end_kpt[1] > 0):
                    
                    # Get limb color from ultralytics palette
                    limb_color_idx = LIMB_COLOR[i]
                    limb_color = POSE_PALETTE[limb_color_idx]
                    # Convert RGB to BGR for OpenCV
                    limb_color_bgr = (limb_color[2], limb_color[1], limb_color[0])
                    
                    start_point = (int(start_kpt[0]), int(start_kpt[1]))
                    end_point = (int(end_kpt[0]), int(end_kpt[1]))
                    
                    cv2.line(image, start_point, end_point, limb_color_bgr, 
                           max(int(line_width / 2), 1), lineType=cv2.LINE_AA)

    # Save result image
    cv2.imwrite(output_path, image)
    print(f"Pose estimation result saved to '{output_path}' file.")

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
    """Process a single image for pose estimation."""
    try:
        # 1. Preprocessing
        input_tensor, original_w, original_h, ratio, pad = preprocess_image(image_path)
        
        # 2. Inference
        outputs = run_inference(ONNX_MODEL_PATH, input_tensor)
        
        # 3. Post-processing
        detections = postprocess_pose(outputs, original_w, original_h, ratio, pad)
        
        # 4. Generate result save path
        filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{filename}_pose.jpg")
        
        # 5. Visualization
        if detections:
            draw_pose(image_path, detections, output_path)
            print(f"[{filename}] Total {len(detections)} persons with poses detected.")
            for i, det in enumerate(detections):
                score = det['score']
                box = det['box']
                keypoints = det['keypoints']
                visible_keypoints = sum(1 for kpt in keypoints if kpt[2] > 0.5)
                print(f"  {i+1}. person: {score:.2f} - Position: ({box[0]}, {box[1]}) ~ ({box[2]}, {box[3]}) - Visible keypoints: {visible_keypoints}/17")
        else:
            # Copy and save original image even if no poses detected
            image = cv2.imread(image_path)
            cv2.imwrite(output_path, image)
            print(f"[{filename}] No poses detected.")
            
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
    
    print(f"Processing {len(image_files)} image files for pose estimation.")
    print(f"Results will be saved in '{OUTPUT_DIR}' folder.")
    print("-" * 50)
    
    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {Path(image_path).name}")
        if process_single_image(image_path, OUTPUT_DIR):
            success_count += 1
    
    print("-" * 50)
    print(f"Pose estimation processing completed: {success_count}/{len(image_files)} successful")
    print(f"Result files saved in '{OUTPUT_DIR}' folder.")