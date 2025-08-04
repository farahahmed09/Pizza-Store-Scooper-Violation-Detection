# --- Imports ---
import cv2
from ultralytics import YOLO
import pika
import numpy as np
import os # NEW: Import os to handle file paths

# --- Paths & Config ---
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\models\yolo12m-v2.pt"
# NEW: Define the path for the output video
OUTPUT_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\outputs\detection_service_output.mp4"
ROIS = [
    (463, 346, 509, 388), #for vid: sah w b3daha ghalt (2)
    (417, 523, 475, 600), #for vid: sah w b3daha ghalt (2)
]

MONITORING_TIMEOUT_SECONDS = 3

# --- Specific IoU thresholds for each interaction type ---
ROI_TRIGGER_THRESHOLD = 0.4
VIOLATION_THRESHOLD = 0.29
SCOOPER_USAGE_THRESHOLD = 0.1

def calculate_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU)"""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0

def get_max_overlap_iou(boxesA, boxesB):
    """Finds the maximum IoU between any box in list A and any box in list B."""
    max_iou = 0.0
    if not boxesA or not boxesB: return max_iou
    for boxA in boxesA:
        for boxB in boxesB:
            iou = calculate_iou(boxA, boxB)
            if iou > max_iou: max_iou = iou
    return max_iou

class Detector:
    """A generic class to detect a specific type of object."""
    def __init__(self, model, target_class_name):
        self.model = model; self.target_class = target_class_name
    def detect(self, results):
        coords = []
        for box in results[0].boxes:
            if self.model.names[int(box.cls)] == self.target_class:
                coords.append(box.xyxy[0])
        return coords

# --- Initialize Models and State ---
model = YOLO(MODEL_PATH)
hand_detector = Detector(model, 'hand')
pizza_detector = Detector(model, 'pizza')
scooper_detector = Detector(model, 'scooper')

# Global state variables that persist between frames
frame_idx = 0
violation_count = 0
hand_was_in_roi_prev_frame = False
grap_it= True
put_it_back = False
system_state = "idle"
monitoring_start_frame = -1
original_fps = 15 
monitoring_timeout_frames = original_fps * MONITORING_TIMEOUT_SECONDS

# NEW: Global variable for the video writer object
video_writer = None

# This function is called for each frame received from RabbitMQ
def process_frame(ch, method, properties, body):
    global frame_idx, violation_count, hand_was_in_roi_prev_frame, put_it_back, system_state, monitoring_start_frame, video_writer , grap_it

    frame_idx += 1

    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # NEW: Initialize the VideoWriter on the very first frame received
    if video_writer is None:
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, original_fps, (width, height))
        print(f"✅ Output video will be saved to {OUTPUT_PATH}", flush=True)

    print(f"\n--- [FRAME {frame_idx}] Received. State: {system_state.upper()} ---", flush=True)

    results = model.track(frame, persist=True, verbose=False)
    hand_coords = hand_detector.detect(results)
    pizza_coords = pizza_detector.detect(results)
    scooper_coords = scooper_detector.detect(results)

    # --- Your Violation Logic (Non-Blocking State Machine) ---
    hand_roi_iou = get_max_overlap_iou(hand_coords, ROIS)
    is_hand_in_roi_now = hand_roi_iou > ROI_TRIGGER_THRESHOLD

    if system_state == "idle":
        print(f"  GRAP IT: {grap_it}", flush=True)
        print(f"  👀PUTTTTT: {put_it_back}", flush=True)
        if is_hand_in_roi_now and not hand_was_in_roi_prev_frame:
            system_state = "monitoring"
            monitoring_start_frame = frame_idx
            print(f"  🔥 TRIGGER! Hand entered ROI. Starting monitoring.", flush=True)
        # elif put_it_back:
        #     print(f"  ✅ Safe scooper usage confirmed. Resetting.", flush=True)
        #     put_it_back = False

    elif system_state == "monitoring":
        hand_pizza_iou = get_max_overlap_iou(hand_coords, pizza_coords)
        scooper_pizza_iou = get_max_overlap_iou(scooper_coords, pizza_coords)
        scooper_hand_iou = get_max_overlap_iou(scooper_coords, hand_coords)
        print(f"  IoU -> Hand-Pizza: {hand_pizza_iou:.2f}, Scooper-Pizza: {scooper_pizza_iou:.2f}, Scooper-Hand: {scooper_hand_iou:.2f}", flush=True)

        is_scooper_in_use = scooper_hand_iou > SCOOPER_USAGE_THRESHOLD or scooper_pizza_iou > SCOOPER_USAGE_THRESHOLD
        print(f"  Scooper In Use Check: {is_scooper_in_use}", flush=True)
        if is_scooper_in_use or grap_it:
            if grap_it==False:
                grap_it=True 

        if is_scooper_in_use:
            print(" ✅ DECISION: Safe scooper usage. Resetting to IDLE.", flush=True)
            system_state = "idle"
        elif hand_pizza_iou > VIOLATION_THRESHOLD and scooper_hand_iou < SCOOPER_USAGE_THRESHOLD and scooper_pizza_iou < SCOOPER_USAGE_THRESHOLD and not grap_it:
            violation_count += 1
            print(f"  🚨 VIOLATION! Hand on pizza without scooper. Total: {violation_count}", flush=True)
            system_state = "idle"
        #elif (frame_idx - monitoring_start_frame) > monitoring_timeout_frames or (is_hand_in_roi_now and not hand_was_in_roi_prev_frame):
        elif (frame_idx - monitoring_start_frame) > monitoring_timeout_frames or (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
            if (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
                #system_state = "idle"
                print("  ⭐ out bec entered again ")
            elif not (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
                print("  🔵 TIMEOUT. Resetting to IDLE.", flush=True)
                system_state = "idle"
                grap_it=False

    hand_was_in_roi_prev_frame = is_hand_in_roi_now

    # --- MODIFIED: Draw annotations and write to video, but don't display ---
    annotated_frame = results[0].plot()
    for x1, y1, x2, y2 in ROIS: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"State: {system_state.upper()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Violations: {violation_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Write the annotated frame to our video file
    if video_writer is not None:
        video_writer.write(annotated_frame)
    
    # REMOVED cv2.imshow() and cv2.waitKey() for background processing

# --- Setup RabbitMQ Consumer ---
try:
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='video_frames')
    channel.basic_consume(queue='video_frames', on_message_callback=process_frame, auto_ack=True)
    
    print('✅ Detection Service started. Waiting for frames...', flush=True)
    channel.start_consuming()
except pika.exceptions.AMQPConnectionError:
    print("❌ Detection Service could not connect to RabbitMQ. Is it running?", flush=True)
except KeyboardInterrupt:
    print("\nStopping consumer...", flush=True)
finally:
    # NEW: Make sure to release the video writer when the script stops
    if video_writer is not None:
        video_writer.release()
        print(f"✅ Output video saved to {OUTPUT_PATH}", flush=True)
    if 'connection' in locals() and connection.is_open:
        connection.close()
    
    # REMOVED cv2.destroyAllWindows()
    print("Detection service stopped.", flush=True)