"""
run_initial_inference.py

This script performs object detection inference on a video file. It loads the
model and displays the results in real-time to visually confirm that all key
objects (Hand, Pizza, Scooper) are detected correctly before implementing the
main violation logic.

- It loads a pretrained YOLOv8 model.
- It reads frames from a video file one by one.
- For each frame, it runs inference to detect objects.
- It uses the model's built-in plotting capabilities to draw bounding boxes.
- The annotated frames are displayed in a window.
- The loop can be exited by pressing the 'q' key.
"""

# --- Imports ---
import cv2
from ultralytics import YOLO

# --- Constants ---
# Make sure to replace 'your_yolo_model.pt' with the actual name of your model file.
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\models\yolo12m-v2.pt"
# Make sure to replace 'Sah w b3dha ghalt.mp4' with the actual name of your video file.
VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\videos\Sah w b3dha ghalt (2).mp4"

def main(model_path: str, video_path: str) -> None:
    """
    Runs inference on a video and displays the results.

    Args:
        model_path (str): The file path to the YOLO model.
        video_path (str): The file path to the video.
    """
    # --- Model Loading ---
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- Video Loading ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at '{video_path}'")
        return

    # --- Main Loop for Video Processing ---
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model.predict(frame)

            # Visualize the results on the frame
            # The .plot() method returns a frame with detections drawn on it
            annotated_frame = results[0].plot()

            # Display the annotated frame in a window
            cv2.imshow("YOLOv12 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            # cv2.waitKey(1) waits 1ms for a key press. It's required for imshow.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo processing finished and resources released.")


if __name__ == "__main__":
    main(model_path=MODEL_PATH, video_path=VIDEO_PATH)