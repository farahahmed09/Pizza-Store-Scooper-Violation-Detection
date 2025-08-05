# --- Imports ---
import cv2
from ultralytics import YOLO
import pika
import numpy as np
import json
import base64
import sqlite3
import datetime
import os
import shutil


# --- Paths & Config ---
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\models\yolo12m-v2.pt"
DB_PATH = "data/violations.db"
VIOLATIONS_DIR = "data/violations" # Directory to save violation files

# Hardcoded regions of interest (ROIs) for detecting hand intrusions
ROIS = [
    (244, 323, 286, 357),
    (272, 200, 307, 242)
]

# Thresholds for detection logic and state transitions
MONITORING_TIMEOUT_SECONDS = 3
ROI_TRIGGER_THRESHOLD = 0.4
VIOLATION_THRESHOLD = 0.29
SCOOPER_USAGE_THRESHOLD = 0.1
WAIT_COUNTER_THRESHOLD = 25

# Calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0

# Return max IoU between two lists of boxes
def get_max_overlap_iou(boxesA, boxesB):
    max_iou = 0.0
    if not boxesA or not boxesB: return max_iou
    for boxA in boxesA:
        for boxB in boxesB:
            iou = calculate_iou(boxA, boxB)
            if iou > max_iou: max_iou = iou
    return max_iou

# Generic detector for specific object class using YOLO
class Detector:
    def __init__(self, model, target_class_name):
        self.model = model; self.target_class = target_class_name

    # Extracts bounding boxes for detected target class
    def detect(self, results):
        coords = []
        for box in results[0].boxes:
            if self.model.names[int(box.cls)] == self.target_class:
                coords.append(box.xyxy[0])
        return coords
# NEW simplified version
def setup_database():
    """Confirms that the violations directory exists."""
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    print("✅ Violations directory confirmed.")

def log_violation(frame_idx, raw_frame, results):
    """Saves violation frame as an image, data as a JSON, and paths to the DB."""
    timestamp = datetime.datetime.now()
    
    # Define file paths
    base_filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S')}_f{frame_idx}"
    image_filename = f"{base_filename}.jpg"
    json_filename = f"{base_filename}.json"
    image_path = os.path.join(VIOLATIONS_DIR, image_filename)
    data_path = os.path.join(VIOLATIONS_DIR, json_filename)

    # 1. Save the raw frame as an image
    cv2.imwrite(image_path, raw_frame)

    # 2. Save the bounding box data as a JSON file
    violation_data = []
    for box in results[0].boxes:
        violation_data.append({
            "class": model.names[int(box.cls)],
            "coordinates": box.xyxy[0].tolist(),
            "confidence": float(box.conf[0])
        })
    with open(data_path, 'w') as f:
        json.dump({"timestamp": timestamp.isoformat(), "frame_index": frame_idx, "detections": violation_data}, f, indent=4)

    # 3. Save the file paths to the database
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

# --- Initialize Models and State ---
model = YOLO(MODEL_PATH)
hand_detector = Detector(model, 'hand')
pizza_detector = Detector(model, 'pizza')
scooper_detector = Detector(model, 'scooper')

# Global state for detection logic across frames
frame_idx = 0
violation_count = 0
hand_was_in_roi_prev_frame = False
grap_it = True  # Indicates whether the hand is currently trying to grab something
system_state = "idle"  # Finite-state machine control
monitoring_start_frame = -1
original_fps = 10
monitoring_timeout_frames = original_fps * MONITORING_TIMEOUT_SECONDS
wait_counter = 0  # Used to avoid false positives from transient violations

# Callback for each frame received via RabbitMQ
def process_frame(ch, method, properties, body):
    global frame_idx, violation_count, hand_was_in_roi_prev_frame, system_state, monitoring_start_frame, grap_it, wait_counter

    frame_idx += 1  # Increment frame index

    # Decode JPEG frame from RabbitMQ body
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print(f"\n--- [FRAME {frame_idx}] Received. State: {system_state.upper()} ---", flush=True)

    # Run tracking inference
    results = model.track(frame, persist=True, verbose=False)

    # Detect hands, pizza, and scooper tools
    hand_coords = hand_detector.detect(results)
    pizza_coords = pizza_detector.detect(results)
    scooper_coords = scooper_detector.detect(results)

    # Compute hand-ROI overlap
    hand_roi_iou = get_max_overlap_iou(hand_coords, ROIS)
    is_hand_in_roi_now = hand_roi_iou > ROI_TRIGGER_THRESHOLD

    # --- FSM: IDLE state ---
    if system_state == "idle":
        print(f"  GRAP IT: {grap_it}", flush=True)
        # Trigger monitoring if hand enters ROI
        if is_hand_in_roi_now and not hand_was_in_roi_prev_frame:
            system_state = "monitoring"
            monitoring_start_frame = frame_idx
            print(f"  🔥 TRIGGER! Hand entered ROI. Starting monitoring.", flush=True)

    # --- FSM: MONITORING state ---
    elif system_state == "monitoring":
        # Calculate key IoUs for logic
        hand_pizza_iou = get_max_overlap_iou(hand_coords, pizza_coords)
        scooper_pizza_iou = get_max_overlap_iou(scooper_coords, pizza_coords)
        scooper_hand_iou = get_max_overlap_iou(scooper_coords, hand_coords)

        print(f"  IoU -> Hand-Pizza: {hand_pizza_iou:.2f}, Scooper-Pizza: {scooper_pizza_iou:.2f}, Scooper-Hand: {scooper_hand_iou:.2f}", flush=True)

        # Check if scooper is being used
        is_scooper_in_use = scooper_hand_iou > SCOOPER_USAGE_THRESHOLD or scooper_pizza_iou > SCOOPER_USAGE_THRESHOLD
        print(f"  Scooper In Use Check: {is_scooper_in_use}", flush=True)

        if is_scooper_in_use or grap_it:
            if grap_it == False:
                grap_it = True  # Reset grap_it flag on safe action

        # Valid scooper use detected
        if is_scooper_in_use:
            print(" ✅ DECISION: Safe scooper usage. Resetting to IDLE.", flush=True)
            system_state = "idle"

        # Violation detected: hand touches pizza without scooper
        elif hand_pizza_iou > VIOLATION_THRESHOLD and scooper_hand_iou < SCOOPER_USAGE_THRESHOLD and scooper_pizza_iou < SCOOPER_USAGE_THRESHOLD and not grap_it:
            wait_counter += 1
            if wait_counter > WAIT_COUNTER_THRESHOLD:
                wait_counter = 0
                violation_count += 1
                print(f"  🚨 VIOLATION! Hand on pizza without scooper. Total: {violation_count}", flush=True)
                # --- NEW: Call the function to save the violation ---
                #annotated_frame_for_db = results[0].plot()
                log_violation(frame_idx, frame, results)
                system_state = "idle"
            else:
                system_state = "monitoring" 

        # Monitoring timeout or hand re-entered ROI without action
        elif (frame_idx - monitoring_start_frame) > monitoring_timeout_frames or (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
            if (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
                print("  ⭐ out bec entered again ")
            elif not (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
                print("  🔵 TIMEOUT. Resetting to IDLE.", flush=True)
                system_state = "idle"
                grap_it = False

    # Update hand presence state for next frame
    hand_was_in_roi_prev_frame = is_hand_in_roi_now

    # Annotate and publish frame to results_queue
    annotated_frame = results[0].plot()
    # --- Prepare and Publish Results as JSON ---
    annotated_frame = results[0].plot()
    for x1, y1, x2, y2 in ROIS: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Encode frame to Base64 text
    success, buffer = cv2.imencode('.jpg', annotated_frame)
    if success:
        frame_as_text = base64.b64encode(buffer).decode('utf-8')
        # Create the structured message
        message = {
            "frame": frame_as_text,
            "violations": violation_count,
            "state": system_state
        }
        # Publish the JSON string
        ch.basic_publish(exchange='', routing_key='results_queue', body=json.dumps(message))
        print(f"Frame {frame_idx}: Published results. Violations: {violation_count}", flush=True)

# --- Setup RabbitMQ Consumer ---
if __name__ == "__main__":
    try:
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='video_frames'); channel.queue_declare(queue='results_queue')
        channel.basic_consume(queue='video_frames', on_message_callback=process_frame, auto_ack=True)
        print('✅ Detection Service started. Waiting for frames...', flush=True)
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\nStopping consumer...", flush=True)
    finally:
        if 'connection' in locals() and connection.is_open: connection.close()
        print("Detection service stopped.", flush=True)
