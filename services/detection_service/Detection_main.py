"""
File: Detection_main.py

Description:
Real-time video processing script that detects food safety violations in 
a pizza store by monitoring if hands touch pizza without using a scooper.
"""

# --- Imports ---
import cv2  # OpenCV for image processing (decoding frames, drawing boxes, saving images).
from ultralytics import YOLO  # YOLOv12 object detection model for identifying hands, pizzas, and scoopers.
import pika  # RabbitMQ client for receiving video frames and sending results via message queues.
import numpy as np  # NumPy for efficient numerical operations, especially for converting byte streams to image arrays.
import json  # For saving detection data in a structured format (JSON) and for messaging.
import base64  # To encode image data into a text format for transmission in JSON payloads.
import sqlite3  # To log violation metadata into a lightweight, file-based SQLite database.
import datetime  # To generate timestamps for logging violations accurately.
import os  # To perform operating system-level tasks like creating directories and managing file paths.
import shutil  # Optional file operations for moving/copying files (imported but not actively used).

# --- Paths & Config ---
# Path to the pre-trained YOLO model file.
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\models\yolo12m-v2.pt"
# Path where the SQLite database for logging violations will be stored.
DB_PATH = "data/violations.db"
# Directory where evidence of violations (images and JSON files) will be saved.
VIOLATIONS_DIR = "data/violations"

# Predefined Regions of Interest (ROIs) where hand intrusions are monitored.
# These are hardcoded as (x1, y1, x2, y2) coordinates for specific areas in the video feed.
ROIS = [
    (244, 323, 286, 357),
    (272, 200, 307, 242)

]

# --- Detection thresholds and timeouts (tuned empirically for this specific use case) ---
MONITORING_TIMEOUT_SECONDS = 3      # How long the system stays in 'monitoring' mode before resetting.
ROI_TRIGGER_THRESHOLD = 0.4         # IoU threshold for a hand to be considered "inside" an ROI.
VIOLATION_THRESHOLD = 0.29          # IoU threshold for hand-pizza overlap to be considered a potential violation.
SCOOPER_USAGE_THRESHOLD = 0.1       # IoU threshold to confirm a scooper is being used with a hand or pizza.
WAIT_COUNTER_THRESHOLD = 15         # Number of consecutive frames a violation must be detected before logging it (debounce).
INFERENCE_FPS = 10                  # Target FPS for running the AI model to optimize performance.

# --- Utility Functions ---

# Compute Intersection over Union (IoU) between two bounding boxes.
def calculate_iou(boxA, boxB):
    """
    Calculates the IoU, a measure of how much two bounding boxes overlap.

    Args:
        boxA (tuple): The first bounding box (x1, y1, x2, y2).
        boxB (tuple): The second bounding box (x1, y1, x2, y2).

    Returns:
        float: The IoU value, ranging from 0.0 (no overlap) to 1.0 (perfect overlap).
    """
    # Determine the (x, y)-coordinates of the intersection rectangle.
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    
    # Compute the area of the intersection rectangle.
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute the area of both bounding boxes.
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute the area of the union.
    union_area = float(boxA_area + boxB_area - intersection_area)
    
    # Return the IoU value; handle division by zero if there's no union area.
    return intersection_area / union_area if union_area > 0 else 0

# Return the maximum IoU value across all box pairs between two lists.
def get_max_overlap_iou(boxesA, boxesB):
    """
    Finds the highest IoU score between any box in one list and any box in another.

    Args:
        boxesA (list): A list of bounding boxes.
        boxesB (list): Another list of bounding boxes.

    Returns:
        float: The maximum IoU value found between any pair of boxes from the two lists.
    """
    max_iou = 0.0
    # If either list is empty, there can be no overlap.
    if not boxesA or not boxesB: return max_iou
    # Iterate through every possible pair of boxes to find the maximum overlap.
    for boxA in boxesA:
        for boxB in boxesB:
            iou = calculate_iou(boxA, boxB)
            if iou > max_iou: max_iou = iou
    return max_iou

# --- Object Detector Wrapper Class ---

class Detector:
    """A simple wrapper class to filter YOLO model results for a specific object class."""
    def __init__(self, model, target_class_name):
        """
        Initializes the detector.

        Args:
            model: The loaded YOLO model instance.
            target_class_name (str): The name of the class to detect (e.g., 'hand', 'pizza').
        """
        self.model = model
        self.target_class = target_class_name

    # Returns bounding boxes of detections matching the target class.
    def detect(self, results):
        """
        Processes raw YOLO results and extracts coordinates for the target class.

        Args:
            results: The output from a YOLO model prediction.

        Returns:
            list: A list of bounding box coordinates [[x1, y1, x2, y2], ...] for the target class.
        """
        coords = []
        # Loop through all detected boxes in the result.
        for box in results[0].boxes:
            # Check if the detected class name matches our target.
            if self.model.names[int(box.cls)] == self.target_class:
                # If it matches, append its coordinates to the list.
                coords.append(box.xyxy[0])
        return coords

# Create the violations directory if it doesn't already exist.
def setup_database():
    """
    Ensures that the directory for storing violation evidence exists.
    Note: The function name is a misnomer; it only sets up the directory, not the DB schema.
    """
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    print("✅ Violations directory confirmed.")

# Logs a violation to disk and the SQLite DB.
def log_violation(frame_idx, raw_frame, results):
    """
    Saves evidence of a violation, including an image and a JSON file with detection details,
    and records the event in the SQLite database.

    Args:
        frame_idx (int): The frame number where the violation occurred.
        raw_frame: The raw OpenCV image frame of the violation.
        results: The YOLO model results for that frame.
    """
    # Get the current time for timestamping.
    timestamp = datetime.datetime.now()
    
    # Define unique filenames and full paths for the image and JSON data.
    base_filename = f"violation_{timestamp.strftime('%Y%m%d_%H%M%S')}_f{frame_idx}"
    image_filename = f"{base_filename}.jpg"
    json_filename = f"{base_filename}.json"
    image_path = os.path.join(VIOLATIONS_DIR, image_filename)
    data_path = os.path.join(VIOLATIONS_DIR, json_filename)

    # Save the raw frame of the violation as a JPEG image.
    cv2.imwrite(image_path, raw_frame)

    # Convert the bounding box results into a structured list of dictionaries for JSON serialization.
    violation_data = []
    for box in results[0].boxes:
        violation_data.append({
            "class": model.names[int(box.cls)],
            "coordinates": box.xyxy[0].tolist(), # Convert tensor to list for JSON.
            "confidence": float(box.conf[0])     # Convert tensor to float.
        })

    # Write the structured data to a JSON file.
    with open(data_path, 'w') as f:
        json.dump({
            "timestamp": timestamp.isoformat(),
            "frame_index": frame_idx,
            "detections": violation_data
        }, f, indent=4)

    # Insert a record of the violation into the SQLite database.
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Execute the SQL INSERT command with parameterized query to prevent SQL injection.
        cursor.execute(
            "INSERT INTO violations (timestamp, frame_index, image_path, data_path) VALUES (?, ?, ?, ?)",
            (timestamp.isoformat(), frame_idx, image_path, data_path)
        )
        conn.commit()  # Commit the transaction to save changes.
        conn.close()   # Close the database connection.
        print(f"  💾 Violation at frame {frame_idx} logged to database.", flush=True)
    except Exception as e:
        # Catch and print any errors that occur during the database operation.
        print(f"❌ Error saving to database: {e}", flush=True)

# --- Model Initialization and Global State ---
# Load the YOLOv12 model from the specified path.
model = YOLO(MODEL_PATH)
# Create specialized detector instances for each class of interest.
hand_detector = Detector(model, 'hand')
pizza_detector = Detector(model, 'pizza')
scooper_detector = Detector(model, 'scooper')

# -- State machine and tracking variables --
frame_idx = 0                       # A counter for incoming frames.
violation_count = 0                 # A running total of confirmed violations.
hand_was_in_roi_prev_frame = False  # A flag to detect when a hand *first* enters an ROI.
grap_it = True                      # A flag indicating if a hand-to-pizza action is currently permissible. Resets after a timeout.
system_state = "idle"               # The current state of the finite state machine (FSM): 'idle' or 'monitoring'.
monitoring_start_frame = -1         # Frame index when the 'monitoring' state began, used for timeouts.
original_fps = 10                   # Assumed FPS of the source video stream.
monitoring_timeout_frames = original_fps * MONITORING_TIMEOUT_SECONDS # Pre-calculate timeout in frames.
wait_counter = 0                    # Counter for the violation debounce logic.
skip_interval = max(1, int(original_fps // INFERENCE_FPS)) # Number of frames to skip between model inferences.
last_results = None                 # Cache for the most recent YOLO model results.

# --- Frame Processing Callback (triggered by RabbitMQ) ---
def process_frame(ch, method, properties, body):
    """
    This function is called for every frame received from the 'video_frames' RabbitMQ queue.
    It contains the core detection logic.
    """
    # Declare which global variables will be modified within this function.
    global frame_idx, violation_count, hand_was_in_roi_prev_frame
    global system_state, monitoring_start_frame, grap_it, wait_counter, last_results

    # Increment the frame counter for tracking.
    frame_idx += 1

    # Decode the received byte stream (JPEG format) into a NumPy array, then into an OpenCV image.
    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print(f"\n--- [FRAME {frame_idx}] Received. State: {system_state.upper()} ---", flush=True)

    # --- OPTIMIZATION: Run the expensive AI model only periodically ---
    # The model is run every `skip_interval` frames to save computational resources.
    if frame_idx % skip_interval == 0:
        # Run YOLO tracking and cache the results. `persist=True` helps track objects across frames.
        last_results = model.track(frame, persist=True, verbose=False)

    # If the model has not run yet (on the very first frames), skip processing.
    if last_results is None:
        return

    # For frames where the model wasn't run, use the most recent cached results for analysis.
    results = last_results

    # Use the detector wrappers to get coordinates for each specific class from the results.
    hand_coords = hand_detector.detect(results)
    pizza_coords = pizza_detector.detect(results)
    scooper_coords = scooper_detector.detect(results)

    # Check if a hand is currently overlapping with any of the predefined ROIs.
    hand_roi_iou = get_max_overlap_iou(hand_coords, ROIS)
    is_hand_in_roi_now = hand_roi_iou > ROI_TRIGGER_THRESHOLD

    # --- FSM: IDLE State ---
    # In the 'idle' state, the system is waiting for a triggering event.
    if system_state == "idle":
        print(f"  GRAP IT: {grap_it}", flush=True)
        # TRIGGER: If a hand enters an ROI (and it wasn't there in the previous frame).
        if is_hand_in_roi_now and not hand_was_in_roi_prev_frame:
            # Transition the state to 'monitoring'.
            system_state = "monitoring"
            # Record the frame number when monitoring started.
            monitoring_start_frame = frame_idx
            print(f"  🔥 TRIGGER! Hand entered ROI. Starting monitoring.", flush=True)

    # --- FSM: MONITORING State ---
    # In the 'monitoring' state, the system actively checks for violations.
    elif system_state == "monitoring":
        # Compute the overlap (IoU) between different detected objects.
        hand_pizza_iou = get_max_overlap_iou(hand_coords, pizza_coords)
        scooper_pizza_iou = get_max_overlap_iou(scooper_coords, pizza_coords)
        scooper_hand_iou = get_max_overlap_iou(scooper_coords, hand_coords)

        print(f"  IoU -> Hand-Pizza: {hand_pizza_iou:.2f}, Scooper-Pizza: {scooper_pizza_iou:.2f}, Scooper-Hand: {scooper_hand_iou:.2f}", flush=True)

        # Check for proper scooper usage (hand holding scooper, or scooper touching pizza).
        is_scooper_in_use = scooper_hand_iou > SCOOPER_USAGE_THRESHOLD or scooper_pizza_iou > SCOOPER_USAGE_THRESHOLD
        print(f"  Scooper In Use Check: {is_scooper_in_use}", flush=True)

        # If a scooper is used correctly, it's a safe action.
        if is_scooper_in_use or grap_it:
            # If `grap_it` was false, reset it to true because a safe action was performed.
            if not grap_it:
                grap_it = True

        # If a scooper is detected, the action is safe. Reset the system to IDLE.
        if is_scooper_in_use:
            print(" ✅ DECISION: Safe scooper usage. Resetting to IDLE.", flush=True)
            system_state = "idle"

        # --- Violation Condition ---
        # A violation occurs if a hand touches a pizza WITHOUT a scooper, and the `grap_it` flag is false.
        elif (
            hand_pizza_iou > VIOLATION_THRESHOLD and
            scooper_hand_iou < SCOOPER_USAGE_THRESHOLD and
            scooper_pizza_iou < SCOOPER_USAGE_THRESHOLD and
            not grap_it
        ):
            # Increment the debounce counter.
            wait_counter += 1
            # If the condition persists for enough frames, confirm and log the violation.
            if wait_counter > WAIT_COUNTER_THRESHOLD:
                wait_counter = 0  # Reset the debounce counter.
                violation_count += 1
                print(f"  🚨 VIOLATION! Hand on pizza without scooper. Total: {violation_count}", flush=True)
                log_violation(frame_idx, frame, results) # Log the violation evidence.
                system_state = "idle" # Reset the state machine.
            else:
                # If debounce threshold is not met, remain in monitoring state.
                system_state = "monitoring"

        # --- Timeout or Re-entry Condition ---
        # Reset to IDLE if monitoring time exceeds the timeout OR if the hand re-enters the ROI.
        elif (frame_idx - monitoring_start_frame) > monitoring_timeout_frames or (hand_roi_iou > ROI_TRIGGER_THRESHOLD):
            if hand_roi_iou > ROI_TRIGGER_THRESHOLD:
                # This message seems to be a reminder for the operator.
                print("  ⭐ put the scooper back after using it")
            else:
                # If it's a timeout, log it and reset the state.
                print("  🔵 TIMEOUT. Resetting to IDLE.", flush=True)
                system_state = "idle"
                # After a timeout, the next grab is no longer "free" and must use a scooper.
                grap_it = False

    # Update the state of hand presence for the next frame's comparison.
    hand_was_in_roi_prev_frame = is_hand_in_roi_now

    # --- Frame Annotation and Publishing ---
    # Draw bounding boxes from the last AI result onto a *copy* of the current frame.
    annotated_frame = last_results[0].plot(img=frame.copy())
    # Draw the predefined ROIs on the frame for visualization.
    for x1, y1, x2, y2 in ROIS:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red rectangle for ROIs.

    # Encode the final annotated frame to JPEG format, then to a Base64 string for JSON transport.
    success, buffer = cv2.imencode('.jpg', annotated_frame)
    if success:
        frame_as_text = base64.b64encode(buffer).decode('utf-8')

        # Create a structured JSON message containing the annotated frame and system status.
        message = {
            "frame": frame_as_text,
            "violations": violation_count,
            "state": system_state
        }
        # Publish the message to the 'results_queue' for a frontend or another service to consume.
        ch.basic_publish(exchange='', routing_key='results_queue', body=json.dumps(message))
        print(f"Frame {frame_idx}: Published results. Violations: {violation_count}", flush=True)

# --- Start RabbitMQ Consumer Loop ---
# This block runs only when the script is executed directly.
if __name__ == "__main__":
    try:
        # Ensure the output directory for violations exists before starting.
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)
        # Establish a connection to the RabbitMQ server running on localhost.
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        # Declare the queue for receiving video frames (ensures it exists).
        channel.queue_declare(queue='video_frames')
        # Declare the queue for sending results (ensures it exists).
        channel.queue_declare(queue='results_queue')
        # Set up the consumer to call the 'process_frame' function for each message in 'video_frames'.
        # `auto_ack=True` automatically acknowledges messages upon receipt.
        channel.basic_consume(queue='video_frames', on_message_callback=process_frame, auto_ack=True)
        
        print('✅ Detection Service started. Waiting for frames...', flush=True)
        # Start the consumer loop. This is a blocking call that waits for messages.
        channel.start_consuming()
        
    except KeyboardInterrupt:
        # Handle graceful shutdown when the user presses Ctrl+C.
        print("\nStopping consumer...", flush=True)
    finally:
        # This block ensures that the connection is closed, even if an error occurs.
        if 'connection' in locals() and connection.is_open:
            connection.close()
        print("Detection service stopped.", flush=True)