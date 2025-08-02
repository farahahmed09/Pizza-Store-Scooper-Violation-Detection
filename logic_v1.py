"""
logic_v1.py

This script introduces the concept of a Region of Interest (ROI) to our
inference pipeline.

- It defines a specific rectangular area (the ROI) to monitor.
- It draws the ROI on the video for visual confirmation.
- It checks if any detected 'hand' enters this ROI and prints a message.
"""

# --- Imports ---
import cv2
from ultralytics import YOLO

# --- Constants ---
MODEL_PATH = "models/your_yolo_model.pt"
VIDEO_PATH = "videos/Sah w b3dha ghalt.mp4"

# --- ROI Definition ---
# Got these coordinates by running get_ROI_coords.py.
# FORMAT: (Top-Left-X, Top-Left-Y, Bottom-Right-X, Bottom-Right-Y)
ROI_COORDS = (371, 254, 533, 713)


def main():
    """
    Main function to run ROI-based hand detection.
    """
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at '{VIDEO_PATH}'")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()

        # --- ROI Visualization and Logic ---
        # Draw the ROI rectangle on the frame
        x1, y1, x2, y2 = ROI_COORDS
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box, thickness 2

        # Loop through each detected object
        for box in results[0].boxes:
            # Check if the detected object is a 'hand'
            if model.names[int(box.cls)] == 'hand':
                # Get the coordinates of the hand's bounding box
                hand_x1, hand_y1, hand_x2, hand_y2 = map(int, box.xyxy[0])
                # Calculate the center of the hand's bounding box
                hand_cx = (hand_x1 + hand_x2) // 2
                hand_cy = (hand_y1 + hand_y2) // 2

                # Check if the hand's center is inside the ROI
                if x1 < hand_cx < x2 and y1 < hand_cy < y2:
                    print("✅ Hand detected inside ROI!")
                    # Draw a green circle at the hand's center for visual confirmation
                    cv2.circle(annotated_frame, (hand_cx, hand_cy), 5, (0, 255, 0), -1)

        cv2.imshow("Violation Detection v1", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nProcessing finished.")


if __name__ == "__main__":
    main()