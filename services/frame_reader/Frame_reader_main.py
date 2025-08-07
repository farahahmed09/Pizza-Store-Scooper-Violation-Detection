"""
File: Frame_reader_main.py

Description:
FastAPI-based web service for:
1. Serving the frontend UI.
2. Streaming real-time detection results via WebSocket (from RabbitMQ).
3. Exposing REST APIs to retrieve violation records and serve saved images.
"""
import cv2
import pika
import time
import os

# --- Configurations ---
# Load video path from environment variable or fallback to default
VIDEO_PATH = os.getenv("VIDEO_PATH", r"D:\ai_projects\Pizza-Store-Scooper-Violation-Detection\data\videos\Sah w b3dha ghalt (2).mp4")
TARGET_FPS = 15  # Frame rate to publish frames at
RESIZE_WIDTH = 1024  # Resize all frames to this width for consistency

# --- RabbitMQ Setup ---
try:
    # Establish connection with local RabbitMQ broker
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    # Declare queue for video frames
    channel.queue_declare(queue='video_frames')
    print("✅ Frame Reader connected to RabbitMQ.")
except pika.exceptions.AMQPConnectionError:
    print("❌ Frame Reader could not connect. Is RabbitMQ running?")
    exit()

# --- OpenCV Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error opening video file: {VIDEO_PATH}")
    exit()

print("🚀 Starting to send frames...")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # For progress tracking

# --- Frame Reading Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video finished. Shutting down frame reader.")
        break

    # --- Resize frame to match target width ---
    h, w, _ = frame.shape
    scale = RESIZE_WIDTH / w
    new_h, new_w = int(h * scale), RESIZE_WIDTH
    resized_frame = cv2.resize(frame, (new_w, new_h))

    # --- Encode frame as JPEG ---
    jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Set JPEG quality to 80%
    success, encoded_image = cv2.imencode('.jpg', resized_frame, jpeg_quality)
    if not success:
        continue  # Skip if encoding fails

    # --- Publish frame to RabbitMQ ---
    channel.basic_publish(
        exchange='',
        routing_key='video_frames',
        body=encoded_image.tobytes()
    )

    # --- Log progress ---
    current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(f"Sent frame {current_frame_num}/{total_frames}")

    # --- Throttle to match target FPS ---
    time.sleep(1 / TARGET_FPS)

# --- Cleanup ---
cap.release()
connection.close()
print("✅ Frame Reader has stopped.")