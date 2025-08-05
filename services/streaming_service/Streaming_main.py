"""
File: Streaming_main.py

Description:
FastAPI-based web service for:
1. Serving the frontend UI.
2. Streaming real-time detection results via WebSocket (from RabbitMQ).
3. Exposing REST APIs to retrieve violation records and serve saved images.
Acts as the interface layer between the backend detection system and the frontend.
"""

# --- Imports ---
import asyncio                  # For async RabbitMQ polling
import pika                     # RabbitMQ client for Python
import uvicorn                  # ASGI server to run FastAPI app
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os                       # File/directory operations
import sqlite3                  # Lightweight local database (for violations)
import json                     # To handle JSON data formats

# --- App and Config Setup ---
app = FastAPI()
DB_PATH = "data/violations.db"            # Path to SQLite database storing violations
VIOLATIONS_DIR = "data/violations"        # Directory where violation images are saved

# --- RabbitMQ Consumer Setup ---
# Connect to local RabbitMQ broker
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the queue where detection results are being published
channel.queue_declare(queue='results_queue')

# Track connected WebSocket clients
active_connections: list[WebSocket] = []

# --- Async RabbitMQ Consumer Loop ---
async def consume_rabbitmq():
    """
    Continuously polls RabbitMQ for new messages from 'results_queue' and 
    broadcasts them to all active WebSocket clients.
    """
    while True:
        method_frame, header_frame, body = channel.basic_get(queue='results_queue')
        if method_frame:
            # Acknowledge message so it's removed from the queue
            channel.basic_ack(method_frame.delivery_tag)
            # Send result to all connected clients
            for connection in active_connections:
                await connection.send_text(body.decode('utf-8'))
        else:
            # Sleep briefly to avoid busy looping
            await asyncio.sleep(0.01)

# --- FastAPI Startup Hook ---
@app.on_event("startup")
async def startup_event():
    """
    Triggered when FastAPI starts. Launches the RabbitMQ consumer task.
    """
    asyncio.create_task(consume_rabbitmq())

# --- WebSocket Route for Video Stream ---
@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """
    Accepts WebSocket connections on /ws/video.
    Keeps the connection open to send real-time detection results.
    """
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keeps connection alive
    except WebSocketDisconnect:
        # Remove client on disconnect
        active_connections.remove(websocket)

# --- Frontend Index Route ---
@app.get("/")
async def get_root():
    """
    Serves the frontend HTML page (index.html).
    """
    html_file_path = 'services/frontend/index.html'
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found</h1>")

# --- API: Fetch Violation Records from SQLite ---
@app.get("/api/violations")
async def get_violations():
    """
    Fetches the list of recorded violations from the SQLite DB, ordered by timestamp.
    Returns list of JSON objects with: id, timestamp, frame index, and image path.
    """
    violations = []

    # If DB doesn't exist, return empty list
    if not os.path.exists(DB_PATH):
        return JSONResponse(content=[])

    try:
        # Connect to SQLite and query violations
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, frame_index, image_path FROM violations ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        # Convert rows to dicts for JSON serialization
        for row in rows:
            violations.append(dict(row))
        return JSONResponse(content=violations)

    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- API: Serve Violation Image Files ---
@app.get("/violations/image/{image_name}")
async def get_violation_image(image_name: str):
    """
    Returns the image file associated with a specific violation.
    Performs basic path sanitization to avoid traversal attacks.
    """
    # Security check: block suspicious paths
    if ".." in image_name or image_name.startswith("/"):
        return JSONResponse(content={"error": "Invalid file path"}, status_code=400)

    # Construct full path and serve if file exists
    file_path = os.path.join(VIOLATIONS_DIR, image_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)

    return JSONResponse(content={"error": "Image not found"}, status_code=404)

# --- Entry Point ---
if __name__ == "__main__":
    # Run FastAPI app using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
