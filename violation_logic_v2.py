# --- Imports ---
import cv2
from ultralytics import YOLO

# --- Paths & Config ---
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\models\yolo12m-v2.pt"
VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\videos\Sah w b3dha ghalt (2).mp4"
ROIS = [
    (463, 346, 509, 388), #for vid: sah w b3daha ghalt (2)
    (417, 523, 475, 600), #for vid: sah w b3daha ghalt (2)
]

MONITORING_TIMEOUT_SECONDS = 3

# --- Specific IoU thresholds for each interaction type ---
# You can now tune each of these values independently.
ROI_TRIGGER_THRESHOLD = 0.5      # How much a hand must overlap with an ROI to trigger monitoring.
VIOLATION_THRESHOLD = 0.28      # How much a hand must overlap with a pizza to count as a violation.
SCOOPER_USAGE_THRESHOLD = 0.1  # How much a scooper must overlap with a hand or pizza to be considered 'in use'.

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

# --- Main Execution ---
if __name__ == "__main__":
    
    # Load the YOLO model once at the start for efficiency.
    model = YOLO(MODEL_PATH)
    # Create a dedicated detector instance for each object class.
    hand_detector = Detector(model, 'hand')
    pizza_detector = Detector(model, 'pizza')
    scooper_detector = Detector(model, 'scooper')
    
    # Open the video file for processing.
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print(f"❌ Error opening video file")
    
    # Get video properties to calculate the timeout in frames.
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    monitoring_timeout_frames = original_fps * MONITORING_TIMEOUT_SECONDS
    
    # --- Initialize State Variables ---
    frame_idx = 0
    violation_count = 0
    # A "memory" flag to help detect when a hand *first* enters an ROI.
    hand_was_in_roi_prev_frame = False
    # A flag to signal that a safe scooper action just occurred.
    put_it_back= False

    # This is the main loop that processes the video frame by frame.
    while cap.isOpened():
        success, frame = cap.read()
        if not success: print("Video ended."); break
        
        frame_idx += 1
        
        # Run the AI model on the current frame to find all objects.
        results = model.track(frame, persist=True, verbose=False)
        # Get the coordinates of hands for the initial ROI check.
        hand_coords = hand_detector.detect(results)
        
        # --- LOGIC ---
        # Calculate the overlap between any hand and any ROI.
        hand_roi_iou = get_max_overlap_iou(hand_coords, ROIS)
        # Determine if a hand is in an ROI in this specific frame.
        is_hand_in_roi_now = hand_roi_iou > ROI_TRIGGER_THRESHOLD

        # TRIGGER CONDITION: This block starts the detailed "monitoring loop".
        # It only runs if a hand is in an ROI now, was NOT in an ROI on the previous frame,
        # and a safe scooper action has not just happened.
        if is_hand_in_roi_now and not hand_was_in_roi_prev_frame and not put_it_back: 
            print(f"--- [FRAME {frame_idx}] TRIGGER! Hand entered ROI. IoU: {hand_roi_iou:.2f}. Entering MONITORING. ---")
            
            # Record the frame number when monitoring starts for the timeout check.
            monitoring_start_frame = frame_idx
            
            # --- Monitoring Loop ---
            # This inner loop takes over to analyze the event until a decision is made.
            while True:
                # It must continue reading and processing new frames.
                success, monitor_frame = cap.read()
                if not success: print("Video ended during monitoring."); break
                
                frame_idx += 1
                print(f"\n- [MONITORING FRAME {frame_idx}] -")

                # Run inference on the new frame within the monitoring loop.
                monitor_results = model.track(monitor_frame, persist=True, verbose=False)
                monitor_hand_coords = hand_detector.detect(monitor_results)
                monitor_pizza_coords = pizza_detector.detect(monitor_results)
                monitor_scooper_coords = scooper_detector.detect(monitor_results)

                # Calculate all relevant overlaps for the decision.
                hand_pizza_iou = get_max_overlap_iou(monitor_hand_coords, monitor_pizza_coords)
                scooper_pizza_iou = get_max_overlap_iou(monitor_scooper_coords, monitor_pizza_coords)
                scooper_hand_iou = get_max_overlap_iou(monitor_scooper_coords, monitor_hand_coords)
                
                print(f"  IoU -> Hand-Pizza: {hand_pizza_iou:.2f}, Scooper-Pizza: {scooper_pizza_iou:.2f}, Scooper-Hand: {scooper_hand_iou:.2f}")

                # Check for a safe action (if a scooper is clearly being used).
                is_scooper_in_use = scooper_hand_iou > SCOOPER_USAGE_THRESHOLD or scooper_pizza_iou > SCOOPER_USAGE_THRESHOLD
                if is_scooper_in_use:
                    print("  DECISION: Safe scooper usage. Exiting monitoring.")
                    # Set the flag to signal that a safe action just happened.
                    put_it_back = True
                    # Exit the monitoring loop.
                    break
                
                # If no scooper is in use, check for a violation.
                elif hand_pizza_iou > VIOLATION_THRESHOLD:
                    print(f"  VIOLATION! Hand on pizza without scooper. IoU: {hand_pizza_iou:.2f}")
                    violation_count += 1
                    print(f"  Total violations: {violation_count}")
                    # Exit the monitoring loop.
                    break
                
                # If no decision is made, check if the monitoring time has run out.
                elif (frame_idx - monitoring_start_frame) > monitoring_timeout_frames:
                    print("  TIMEOUT. Exiting monitoring.")
                    # Exit the monitoring loop.
                    break

                # Display the frame while inside the monitoring loop.
                annotated_frame = monitor_results[0].plot()
                cv2.putText(annotated_frame, "STATE: MONITORING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Violations: {violation_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Detection System", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # This handles the 'q' key press to exit the monitoring loop.
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            print("--- Exited MONITORING. Returning to IDLE. ---")

        # This 'else' block runs on every frame that is NOT a trigger.
        else:
            # Its job is to check the 'put_it_back' flag and reset it.
            if put_it_back:
                print(f"--- [FRAME {frame_idx}] DECISION: Safe scooper usage detected. Returning to IDLE. ---")
                put_it_back = False
        
        # --- Display the IDLE frame ---
        # This section displays the output when the system is in the main 'idle' state.
        annotated_frame = results[0].plot()
        # Draw the ROI boxes on the frame.
        for x1, y1, x2, y2 in ROIS:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.putText(annotated_frame, "STATE: IDLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Violations: {violation_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Detection System", annotated_frame)

        # Update the 'memory' flag for the next frame's trigger check.
        hand_was_in_roi_prev_frame = is_hand_in_roi_now

        # This handles the 'q' key press to exit the main loop.
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Release resources when the loop is finished.
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Script finished. Total violations: {violation_count}")