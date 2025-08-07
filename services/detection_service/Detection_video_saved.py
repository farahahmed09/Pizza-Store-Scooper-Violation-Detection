"""
File: Detection_video_saved.py

Description:
Video processing script that detects food safety violations in 
a pizza store by monitoring if hands touch pizza without using a scooper.
This version can read from an input video file OR a live webcam feed.
If reading from a file, it saves an annotated output video.
"""

# --- Imports ---
import cv2                    # OpenCV for image and video processing
from ultralytics import YOLO  # YOLOv12 object detection model
import numpy as np            # NumPy for array and matrix operations
import json                   # For saving detection data in structured format
import sqlite3                # To log violations into SQLite database
import datetime               # To timestamp violations
import os                     # File system operations
import sys                    # For system-level operations like exiting

# --- Paths & Config ---
INPUT_VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\videos\Sah w b3dha ghalt (2).mp4" # <--- SET TO 0 FOR WEBCAM
OUTPUT_VIDEO_DIR = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\output_vid"
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\models\yolo12m-v2.pt"
DB_PATH = "data/violations.db"
VIOLATIONS_DIR = "data/violations"

# Predefined regions where hand intrusions are monitored (hardcoded ROIs)
ROIS = [

    (452, 336, 521, 385),
    (417, 587, 475, 531)
]

# Detection thresholds and timeouts (tuned empirically)
MONITORING_TIMEOUT_SECONDS = 3
ROI_TRIGGER_THRESHOLD = 0.4
VIOLATION_THRESHOLD = 0.29
SCOOPER_USAGE_THRESHOLD = 0.1
WAIT_COUNTER_THRESHOLD = 15
INFERENCE_FPS = 10

# --- Utility Functions ---

def calculate_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0

def get_max_overlap_iou(boxesA, boxesB):
    """Return the maximum IoU value across all box pairs between two lists."""
    max_iou = 0.0
    if not boxesA or not boxesB: return max_iou
    for boxA in boxesA:
        for boxB in boxesB:
            iou = calculate_iou(boxA, boxB)
            if iou > max_iou: max_iou = iou
    return max_iou

# --- Object Detector Wrapper Class ---

class Detector:
    def __init__(self, model, target_class_name):
        self.model = model
        self.target_class = target_class_name

    def detect(self, results):
        """Returns bounding boxes of detections matching the target class."""
        coords = []
        for box in results[0].boxes:
            if self.model.names[int(box.cls)] == self.target_class:
                coords.append(box.xyxy[0])
        return coords

# --- Setup and Logging ---

def setup_directories():
    """Create the violations and output directories if they don't already exist."""
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True) # <<< CHANGE 2: Use OUTPUT_VIDEO_DIR
    print("✅ Output and violations directories are ready.")

def log_violation(frame_idx, raw_frame, results):
    """Logs a violation to disk (image/JSON) and the SQLite DB."""
    timestamp = datetime.datetime.now()
    
    base_filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S')}_f{frame_idx}"
    image_filename = f"{base_filename}.jpg"
    json_filename = f"{base_filename}.json"
    image_path = os.path.join(VIOLATIONS_DIR, image_filename)
    data_path = os.path.join(VIOLATIONS_DIR, json_filename)

    cv2.imwrite(image_path, raw_frame)

    violation_data = []
    for box in results[0].boxes:
        violation_data.append({
            "class": model.names[int(box.cls)],
            "coordinates": box.xyxy[0].tolist(),
            "confidence": float(box.conf[0])
        })

    with open(data_path, 'w') as f:
        json.dump({
            "timestamp": timestamp.isoformat(),
            "frame_index": frame_idx,
            "detections": violation_data
        }, f, indent=4)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO violations (timestamp, frame_index, image_path, data_path) VALUES (?, ?, ?, ?)",
            (timestamp.isoformat(), frame_idx, image_path, data_path)
        )
        conn.commit()
        conn.close()
        print(f"       💾 Violation at frame {frame_idx} logged to database.", flush=True)
    except Exception as e:
        print(f"       ❌ Error saving to database: {e}", flush=True)

# --- Model Initialization ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading YOLO model from '{MODEL_PATH}': {e}")
    sys.exit(1)
    
hand_detector = Detector(model, 'hand')
pizza_detector = Detector(model, 'pizza')
scooper_detector = Detector(model, 'scooper')

# --- Global State Variables ---
violation_count = 0
hand_was_in_roi_prev_frame = False
grap_it = True
system_state = "idle"
monitoring_start_frame = -1
wait_counter = 0
last_results = None
monitoring_timeout_frames = 0 # Will be initialized in main
skip_interval = 1 # Will be initialized in main


def run_detection_logic(frame, frame_idx):
    """
    Processes a single video frame to detect violations.
    Modifies global state variables and returns the annotated frame.
    """
    global violation_count, hand_was_in_roi_prev_frame
    global system_state, monitoring_start_frame, grap_it, wait_counter, last_results

    print(f"\n--- [FRAME {frame_idx}] State: {system_state.upper()} ---", flush=True)

    # --- Optimization: Run AI model only periodically ---
    if frame_idx % skip_interval == 0:
        last_results = model.track(frame, persist=True, verbose=False)

    if last_results is None:
        return frame # Return original frame if no inference has run yet

    # Use the most recent AI results for detection
    results = last_results

    # Run class-specific detections
    hand_coords = hand_detector.detect(results)
    pizza_coords = pizza_detector.detect(results)
    scooper_coords = scooper_detector.detect(results)

    # Check if hand overlaps with any defined ROIs
    hand_roi_iou = get_max_overlap_iou(hand_coords, ROIS)
    is_hand_in_roi_now = hand_roi_iou > ROI_TRIGGER_THRESHOLD

    # --- FSM: IDLE State ---
    if system_state == "idle":
        print(f"  GRAP IT: {grap_it}", flush=True)
        if is_hand_in_roi_now and not hand_was_in_roi_prev_frame:
            system_state = "monitoring"
            monitoring_start_frame = frame_idx
            print(f"  🔥 TRIGGER! Hand entered ROI. Starting monitoring.", flush=True)

    # --- FSM: MONITORING State ---
    elif system_state == "monitoring":
        # Compute IoU relationships between objects
        hand_pizza_iou = get_max_overlap_iou(hand_coords, pizza_coords)
        scooper_pizza_iou = get_max_overlap_iou(scooper_coords, pizza_coords)
        scooper_hand_iou = get_max_overlap_iou(scooper_coords, hand_coords)

        print(f"  IoU -> Hand-Pizza: {hand_pizza_iou:.2f}, Scooper-Pizza: {scooper_pizza_iou:.2f}, Scooper-Hand: {scooper_hand_iou:.2f}", flush=True)

        # Check for proper scooper usage
        is_scooper_in_use = scooper_hand_iou > SCOOPER_USAGE_THRESHOLD or scooper_pizza_iou > SCOOPER_USAGE_THRESHOLD
        print(f"  Scooper In Use Check: {is_scooper_in_use}", flush=True)

        if is_scooper_in_use or grap_it:
            if not grap_it:
                grap_it = True  # Safe usage resets the grab state

        if is_scooper_in_use:
            print(" ✅ DECISION: Safe scooper usage. Resetting to IDLE.", flush=True)
            system_state = "idle"

        # --- Violation Condition ---
        elif (
            hand_pizza_iou > VIOLATION_THRESHOLD and
            scooper_hand_iou < SCOOPER_USAGE_THRESHOLD and
            scooper_pizza_iou < SCOOPER_USAGE_THRESHOLD and
            not grap_it
        ):
            wait_counter += 1
            if wait_counter > WAIT_COUNTER_THRESHOLD:
                wait_counter = 0
                violation_count += 1
                print(f"  🚨 VIOLATION! Hand on pizza without scooper. Total: {violation_count}", flush=True)
                log_violation(frame_idx, frame, results)
                system_state = "idle"
            else:
                system_state = "monitoring"

        # Timeout or user re-entered ROI again
        elif (frame_idx - monitoring_start_frame) > monitoring_timeout_frames or (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
            if hand_roi_iou > ROI_TRIGGER_THRESHOLD:
                print("  ⭐ put the scooper back after using it")
            else:
                print("  🔵 TIMEOUT. Resetting to IDLE.", flush=True)
                system_state = "idle"
                grap_it = False

    hand_was_in_roi_prev_frame = is_hand_in_roi_now

    # --- Annotation and Display ---
    annotated_frame = frame.copy()
    if last_results:
        annotated_frame = last_results[0].plot(img=annotated_frame)

    for x1, y1, x2, y2 in ROIS:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    status_text = f"Status: {system_state.upper()}"
    violation_text = f"Violations: {violation_count}"
    
    cv2.rectangle(annotated_frame, (5, 5), (280, 65), (0, 0, 0), -1) 
    cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    violation_color = (0, 0, 255) if violation_count > 0 else (0, 255, 0)
    cv2.putText(annotated_frame, violation_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, violation_color, 2)
    
    return annotated_frame

# <<< CHANGE 3: MAIN EXECUTION BLOCK NOW HANDLES WEBCAM AND FILE INPUT >>>
if __name__ == "__main__":
    setup_directories()

    # --- Video Input ---
    using_webcam = False
    if INPUT_VIDEO_PATH == 0 or not os.path.exists(INPUT_VIDEO_PATH):
        if INPUT_VIDEO_PATH != 0:
            print(f"⚠️ Warning: Video file not found at '{INPUT_VIDEO_PATH}'.")
        print("✅ Switching to webcam feed...")
        cap = cv2.VideoCapture(0) # Use 0 for default webcam
        using_webcam = True
    else:
        print(f"✅ Opening video file: {INPUT_VIDEO_PATH}")
        cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source. Please check camera connection or file path.")
        sys.exit(1)

    # --- Video Output (only for file processing) ---
    out = None
    if not using_webcam:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        monitoring_timeout_frames = original_fps * MONITORING_TIMEOUT_SECONDS
        skip_interval = max(1, int(original_fps // INFERENCE_FPS))
        
        # Create a valid output file path
        base_filename = os.path.basename(INPUT_VIDEO_PATH)
        output_filename = os.path.join(OUTPUT_VIDEO_DIR, f"output_{base_filename}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, original_fps, (frame_width, frame_height))
        print(f"🎞️  Annotated video will be saved to: {output_filename}")
    else:
        # For webcam, use default values
        monitoring_timeout_frames = INFERENCE_FPS * MONITORING_TIMEOUT_SECONDS
        skip_interval = 2 # Process every other frame for smoother webcam feed

    # --- Main Loop ---
    frame_idx = 0
    print("🚀 Starting detection... Press 'q' in the display window to quit.")
    
    while True: # Changed from cap.isOpened() to allow breaking from inside
        ret, frame = cap.read()
        if not ret:
            if not using_webcam:
                print("✅ End of video file reached.")
            else:
                print("❌ Error: Could not read frame from webcam.")
            break

        frame_idx += 1
        
        annotated_frame = run_detection_logic(frame, frame_idx)

        # Write frame to output file if processing a video
        if out is not None:
            out.write(annotated_frame)
        
        cv2.imshow('Pizza Store Violation Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 User pressed 'q'. Exiting.")
            break

    # --- Release Resources ---
    print("\n✅ Processing finished. Releasing resources.")
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()