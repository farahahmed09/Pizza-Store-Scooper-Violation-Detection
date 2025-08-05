"""
File: Detection_main.py

Description:
Real-time video processing script that detects food safety violations in 
a pizza store by monitoring if hands touch pizza without using a scooper.
"""

# --- Imports ---
import cv2  # OpenCV for image processing
from ultralytics import YOLO  # YOLOv12 object detection model
import pika  # RabbitMQ client for receiving/sending frames
import numpy as np  # NumPy for array and matrix operations
import json  # For saving detection data in structured format
import base64  # To encode image data for publishing
import sqlite3  # To log violations into SQLite database
import datetime  # To timestamp violations
import os  # File system operations
import shutil  # Optional file operations (not used here)

# --- Paths & Config ---
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\models\yolo12m-v2.pt"
DB_PATH = "data/violations.db"  # SQLite DB path for logging violations
VIOLATIONS_DIR = "data/violations"  # Directory to save violation images and JSONs

# Predefined regions where hand intrusions are monitored (hardcoded ROIs)
ROIS = [
    (244, 323, 286, 357),
    (272, 200, 307, 242)
]

# Detection thresholds and timeouts (tuned empirically)
MONITORING_TIMEOUT_SECONDS = 3
ROI_TRIGGER_THRESHOLD = 0.4
VIOLATION_THRESHOLD = 0.29
SCOOPER_USAGE_THRESHOLD = 0.1
WAIT_COUNTER_THRESHOLD = 15

# --- Utility Functions ---

# Compute Intersection over Union (IoU) between two bounding boxes
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0

# Return the maximum IoU value across all box pairs between two lists
def get_max_overlap_iou(boxesA, boxesB):
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

    # Returns bounding boxes of detections matching the target class
    def detect(self, results):
        coords = []
        for box in results[0].boxes:
            if self.model.names[int(box.cls)] == self.target_class:
                coords.append(box.xyxy[0])
        return coords

# Create the violations directory if it doesn't already exist
def setup_database():
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    print("✅ Violations directory confirmed.")

# Logs a violation to disk and the SQLite DB
def log_violation(frame_idx, raw_frame, results):
    timestamp = datetime.datetime.now()
    
    # Define filenames and paths for saving image and JSON
    base_filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S')}_f{frame_idx}"
    image_filename = f"{base_filename}.jpg"
    json_filename = f"{base_filename}.json"
    image_path = os.path.join(VIOLATIONS_DIR, image_filename)
    data_path = os.path.join(VIOLATIONS_DIR, json_filename)

    # Save the violation frame as JPEG image
    cv2.imwrite(image_path, raw_frame)

    # Convert bounding box results to structured JSON
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

    # Insert violation metadata into the database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO violations (timestamp, frame_index, image_path, data_path) VALUES (?, ?, ?, ?)",
            (timestamp.isoformat(), frame_idx, image_path, data_path)
        )
        conn.commit()
        conn.close()
        print(f"  💾 Violation at frame {frame_idx} logged to database.", flush=True)
    except Exception as e:
        print(f"❌ Error saving to database: {e}", flush=True)

# --- Model Initialization and Global State ---
model = YOLO(MODEL_PATH)
hand_detector = Detector(model, 'hand')
pizza_detector = Detector(model, 'pizza')
scooper_detector = Detector(model, 'scooper')

frame_idx = 0
violation_count = 0
hand_was_in_roi_prev_frame = False
grap_it = True  # Flag: whether current hand attempt is allowed
system_state = "idle"  # FSM: 'idle' or 'monitoring'
monitoring_start_frame = -1
original_fps = 10
monitoring_timeout_frames = original_fps * MONITORING_TIMEOUT_SECONDS
wait_counter = 0  # Prevent logging temporary/premature violations

# --- Frame Processing Callback (triggered by RabbitMQ) ---
def process_frame(ch, method, properties, body):
    global frame_idx, violation_count, hand_was_in_roi_prev_frame
    global system_state, monitoring_start_frame, grap_it, wait_counter

    frame_idx += 1  # Track current frame number

    # Decode JPEG frame to OpenCV format
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print(f"\n--- [FRAME {frame_idx}] Received. State: {system_state.upper()} ---", flush=True)

    # Run YOLOv8 object tracking
    results = model.track(frame, persist=True, verbose=False)

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

    # Update hand ROI presence state for next frame comparison
    hand_was_in_roi_prev_frame = is_hand_in_roi_now

    # Annotate frame with detections and ROIs
    annotated_frame = results[0].plot()
    for x1, y1, x2, y2 in ROIS:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Encode final frame as JPEG -> Base64
    success, buffer = cv2.imencode('.jpg', annotated_frame)
    if success:
        frame_as_text = base64.b64encode(buffer).decode('utf-8')

        # Send structured result over message queue
        message = {
            "frame": frame_as_text,
            "violations": violation_count,
            "state": system_state
        }
        ch.basic_publish(exchange='', routing_key='results_queue', body=json.dumps(message))
        print(f"Frame {frame_idx}: Published results. Violations: {violation_count}", flush=True)

# --- Start RabbitMQ Consumer Loop ---
if __name__ == "__main__":
    try:
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)  # Ensure output dir exists
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='video_frames')
        channel.queue_declare(queue='results_queue')
        channel.basic_consume(queue='video_frames', on_message_callback=process_frame, auto_ack=True)
        print('✅ Detection Service started. Waiting for frames...', flush=True)
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\nStopping consumer...", flush=True)
    finally:
        if 'connection' in locals() and connection.is_open:
            connection.close()
        print("Detection service stopped.", flush=True)
