"""
File: Frame_reader_main.py

Description:
FastAPI-based web service for:
1. Serving the frontend UI.
2. Streaming real-time detection results via WebSocket (from RabbitMQ).
3. Exposing REST APIs to retrieve violation records and serve saved images.
"""
# NOTE: The code below implements a video frame publisher (a "producer"). 
# It reads a video file and sends its frames to a RabbitMQ queue. 
# It is not a FastAPI web service as described above.

import cv2      # OpenCV for video file reading, frame resizing, and image encoding.
import pika     # RabbitMQ client library for publishing messages (frames) to a queue.
import time     # Used for throttling the frame sending rate to a target FPS.
import os       # Used to access environment variables for configuration.

# --- Configurations ---
# Load the video file path from a system environment variable named 'VIDEO_PATH'.
# If the environment variable is not set, it falls back to a default hardcoded path.
VIDEO_PATH = os.getenv("VIDEO_PATH", r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\videos\Sah w b3dha ghalt (2).mp4")
TARGET_FPS = 30  # The desired frame rate to publish frames at, simulating a real-time camera feed.
RESIZE_WIDTH = 1024  # All frames will be resized to this width to ensure consistent input for the detection model.

# --- RabbitMQ Setup ---
try:
    # Establish a synchronous blocking connection with the RabbitMQ server running on the local machine.
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    # Create a channel, which is the primary way to interact with the broker.
    channel = connection.channel()
    
    # Declare a queue named 'video_frames'. This is idempotent; it will create the queue if it doesn't exist.
    # This ensures the producer can start even before the consumer is ready.
    channel.queue_declare(queue='video_frames')
    print("✅ Frame Reader connected to RabbitMQ.")
# Handle connection errors gracefully if the RabbitMQ server is not running or is unreachable.
except pika.exceptions.AMQPConnectionError:
    print("❌ Frame Reader could not connect. Is RabbitMQ running?")
    exit() # Exit the script if a connection cannot be made.

# --- OpenCV Video Setup ---
# Create a VideoCapture object to read from the specified video file.
cap = cv2.VideoCapture(VIDEO_PATH)
# Check if the video file was opened successfully.
if not cap.isOpened():
    print(f"❌ Error opening video file: {VIDEO_PATH}")
    exit() # Exit if the video file is invalid or not found.

print("🚀 Starting to send frames...")
# Get the total number of frames in the video for progress tracking purposes.
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Frame Reading Loop ---
# Loop continues as long as the VideoCapture object is open and has frames to read.
while cap.isOpened():
    # Read a single frame from the video. 'success' is a boolean, 'frame' is the image data.
    success, frame = cap.read()
    # If 'success' is False, it means the end of the video has been reached.
    if not success:
        print("Video finished. Shutting down frame reader.")
        break # Exit the loop.

    # --- Resize frame to match target width ---
    # Get the original height and width of the frame.
    h, w, _ = frame.shape
    # Calculate the scaling factor based on the target width.
    scale = RESIZE_WIDTH / w
    # Calculate the new dimensions while maintaining the original aspect ratio.
    new_h, new_w = int(h * scale), RESIZE_WIDTH
    # Perform the resize operation.
    resized_frame = cv2.resize(frame, (new_w, new_h))

    # --- Encode frame as JPEG ---
    # Set the JPEG compression quality (0-100). 80 is a good balance of quality and size.
    jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    # Encode the resized frame into the JPEG format in memory. 'encoded_image' is a NumPy array of bytes.
    success, encoded_image = cv2.imencode('.jpg', resized_frame, jpeg_quality)
    # If encoding fails for any reason, skip this frame.
    if not success:
        continue

    # --- Publish frame to RabbitMQ ---
    # Publish the message to the queue.
    channel.basic_publish(
        exchange='',                # Use the default exchange, which routes messages directly to a queue.
        routing_key='video_frames', # The name of the queue to send the message to.
        body=encoded_image.tobytes()# The message payload: the JPEG data converted to a byte string.
    )

    # --- Log progress ---
    # Get the current frame number from the video capture object.
    current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # Print a status message to the console.
    print(f"Sent frame {current_frame_num}/{total_frames}")

    # --- Throttle to match target FPS ---
    # Pause the loop for a short duration to control the rate of sending frames.
    # This prevents overwhelming the consumer and simulates a live feed at TARGET_FPS.
    time.sleep(1 / TARGET_FPS)

# --- Cleanup ---
# Release the video capture object, freeing the video file.
cap.release()
# Close the connection to the RabbitMQ server.
connection.close()
print("✅ Frame Reader has stopped.")