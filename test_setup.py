"""
test_setup.py

This script verifies the project setup by:
1. Loading the YOLO object detection model.
2. Opening the specified video file.
3. Reading the first frame and running a single inference.

This confirms that all necessary libraries are installed and file paths are correct.
"""

# --- Imports ---
import cv2
from ultralytics import YOLO

# --- Constants ---
# Make sure to replace 'your_yolo_model.pt' with the actual name of your model file.
MODEL_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\models\yolo12m-v2.pt"
# Make sure to replace 'Sah w b3dha ghalt.mp4' with the actual name of your video file.
VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\videos\Sah w b3dha ghalt.mp4"

def main(model_path: str, video_path: str) -> None:
    """
    Main function to load model, video, and test inference.

    Args:
        model_path (str): The file path to the YOLO model.
        video_path (str): The file path to the video.
    """
    # --- Model Loading ---
    print("Loading model...")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return # Exit if the model can't be loaded

    # --- Video Loading ---
    print("\nLoading video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at '{video_path}'")
        return # Exit if the video can't be opened

    print("✅ Video loaded successfully!")

    # --- Frame Reading and Inference ---
    success, frame = cap.read()
    if success:
        print("✅ First frame read successfully!")

        print("\nRunning inference on the first frame...")
        # The 'verbose=False' argument hides detailed per-frame output
        results = model.predict(frame, verbose=False)
        print("✅ Inference complete!")
        print("\n🎉 --- Setup Test Successful! --- 🎉")
    else:
        print("❌ Error: Could not read the first frame from the video.")

    # --- Cleanup ---
    cap.release()
    print("\nResources released.")


if __name__ == "__main__":
    # This block runs only when the script is executed directly
    # (not when it's imported by another script)
    main(model_path=MODEL_PATH, video_path=VIDEO_PATH)