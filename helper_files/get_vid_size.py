"""
get_video_size.py

A simple utility script to get the dimensions (width and height) of a video file.
"""

# --- Imports ---
import cv2
import os

# --- Configuration ---
# IMPORTANT: Replace this with the path to your video file.
# Using a raw string (r"...") is recommended on Windows to avoid issues with backslashes.
VIDEO_PATH = r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\videos\Sah w b3dha ghalt (2).mp4"

def get_video_dimensions(video_path):
    """
    Opens a video file and prints its width and height.

    Args:
        video_path (str): The full path to the video file.
    """
    # First, check if the file exists at the given path
    if not os.path.exists(video_path):
        print(f"❌ ERROR: The file was not found at the path: {video_path}")
        return

    # Open the video file using OpenCV's VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open or read the video file. It might be corrupted or in an unsupported format.")
        return

    # Get the width and height of the video frames
    # cv2.CAP_PROP_FRAME_WIDTH corresponds to the integer code 3
    # cv2.CAP_PROP_FRAME_HEIGHT corresponds to the integer code 4
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get the total number of frames for extra information
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object to free up resources
    cap.release()
    
    # Print the results
    print("\n--- Video Details ---")
    print(f"File Path: {video_path}")
    print(f"✅ Dimensions: {width} x {height} (Width x Height)")
    print(f"Total Frames: {frame_count}")


if __name__ == "__main__":
    get_video_dimensions(VIDEO_PATH)