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
import asyncio              # For running asynchronous background tasks, like the RabbitMQ consumer.
import pika                 # The Python client library for RabbitMQ.
import uvicorn              # The ASGI server used to run the FastAPI application.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse # Specific response types for web endpoints.
import os                   # For operating system interactions like checking file paths.
import sqlite3              # For connecting to the SQLite database to read violation logs.
import json                 # To handle data in JSON format, primarily for API responses.

# --- App and Config Setup ---
# Create an instance of the FastAPI application. This is the main entry point for the web service.
app = FastAPI()
# Define a constant for the path to the SQLite database file.
DB_PATH = "data/violations.db"
# Define a constant for the directory where captured violation images are stored.
VIOLATIONS_DIR = "data/violations"

# --- RabbitMQ Consumer Setup ---
# Establish a blocking connection to the RabbitMQ server running on the local machine.
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
# Create a communication channel with the RabbitMQ broker.
channel = connection.channel()

# Declare the queue from which this service will consume messages.
# This ensures the 'results_queue' exists. The detection script publishes to this queue.
channel.queue_declare(queue='results_queue')

# A list to keep track of all currently active WebSocket connections.
# This allows broadcasting messages to all connected clients.
active_connections: list[WebSocket] = []

# --- Async RabbitMQ Consumer Loop ---
async def consume_rabbitmq():
    """
    Continuously polls RabbitMQ for new messages from 'results_queue' and 
    broadcasts them to all active WebSocket clients. This runs as a background task.
    """
    # This infinite loop continuously checks for new messages.
    while True:
        # Fetch a single message from the queue in a non-blocking manner.
        method_frame, header_frame, body = channel.basic_get(queue='results_queue')
        
        # Check if a message was successfully retrieved.
        if method_frame:
            # Acknowledge the message. This tells RabbitMQ that the message has been processed
            # and can be safely removed from the queue. This prevents message loss on consumer crash.
            channel.basic_ack(method_frame.delivery_tag)
            
            # Broadcast the received message body to every connected WebSocket client.
            for connection in active_connections:
                await connection.send_text(body.decode('utf-8')) # Decode bytes to string for sending.
        else:
            # If the queue is empty, pause briefly to prevent a "busy-wait" loop
            # that would consume 100% CPU. This yields control to the asyncio event loop.
            await asyncio.sleep(0.01)

# --- FastAPI Startup Hook ---
@app.on_event("startup")
async def startup_event():
    """
    This function is automatically triggered once when the FastAPI application starts up.
    It's used here to launch the RabbitMQ consumer as a background task.
    """
    # Schedule the `consume_rabbitmq` function to run concurrently with the web server.
    asyncio.create_task(consume_rabbitmq())

# --- WebSocket Route for Video Stream ---
@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """
    This endpoint handles WebSocket connections at '/ws/video'. It accepts new clients,
    adds them to the list of active connections, and keeps the connection open
    to stream real-time detection results from the RabbitMQ consumer.
    """
    # Accept the incoming WebSocket connection.
    await websocket.accept()
    # Add the new connection to our list of active clients.
    active_connections.append(websocket)
    try:
        # Loop indefinitely to keep the connection alive.
        # `receive_text` waits for a message from the client, effectively pausing execution here.
        while True:
            await websocket.receive_text()
    # This exception is raised when the client disconnects.
    except WebSocketDisconnect:
        # On disconnect, remove the client from the list of active connections
        # to prevent trying to send messages to a closed connection.
        active_connections.remove(websocket)

# --- Frontend Index Route ---
@app.get("/")
async def get_root():
    """
    Serves the main frontend page (index.html) when a user visits the root URL.
    """
    html_file_path = 'services/frontend/index.html'
    # Check if the HTML file exists at the specified path.
    if os.path.exists(html_file_path):
        # If it exists, open, read, and return its content as an HTML response.
        with open(html_file_path, 'r') as f:
            return HTMLResponse(f.read())
    # If the file is not found, return a simple error message.
    return HTMLResponse("<h1>index.html not found</h1>")

# --- API: Fetch Violation Records from SQLite ---
@app.get("/api/violations")
async def get_violations():
    """
    This API endpoint fetches all recorded violations from the SQLite database.
    It returns a JSON array of violation records, ordered from newest to oldest.
    """
    violations = []

    # If the database file doesn't exist yet, return an empty list immediately.
    if not os.path.exists(DB_PATH):
        return JSONResponse(content=[])

    try:
        # Connect to the SQLite database.
        conn = sqlite3.connect(DB_PATH)
        # Set the row factory to sqlite3.Row to get dictionary-like access to columns.
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Execute a SQL query to select all relevant fields from the violations table.
        cursor.execute("SELECT id, timestamp, frame_index, image_path FROM violations ORDER BY timestamp DESC")
        # Fetch all resulting rows from the query.
        rows = cursor.fetchall()
        # Close the database connection.
        conn.close()

        # Convert each `sqlite3.Row` object into a standard Python dictionary.
        for row in rows:
            violations.append(dict(row))
        # Return the list of violations as a JSON response.
        return JSONResponse(content=violations)

    # Catch any potential database errors.
    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")
        # Return a 500 Internal Server Error response with the error message.
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- API: Serve Violation Image Files ---
@app.get("/violations/image/{image_name}")
async def get_violation_image(image_name: str):
    """
    This API endpoint serves a specific violation image file.
    The `image_name` is provided as a path parameter in the URL.
    """
    # --- Security Check ---
    # Prevent directory traversal attacks by checking for ".." or absolute paths.
    # This ensures a user cannot request files outside of the intended `VIOLATIONS_DIR`.
    if ".." in image_name or image_name.startswith("/"):
        return JSONResponse(content={"error": "Invalid file path"}, status_code=400)

    # Safely construct the full path to the requested image file.
    file_path = os.path.join(VIOLATIONS_DIR, image_name)
    
    # Check if the file actually exists at the constructed path.
    if os.path.exists(file_path):
        # If it exists, return the file as a FileResponse, which efficiently streams it.
        return FileResponse(file_path)

    # If the file does not exist, return a 404 Not Found error.
    return JSONResponse(content={"error": "Image not found"}, status_code=404)

# --- Entry Point ---
# This block executes only when the script is run directly (e.g., `python Streaming_main.py`).
if __name__ == "__main__":
    # Start the Uvicorn ASGI server to run our FastAPI application.
    # `host="0.0.0.0"` makes the server accessible on the network, not just from the local machine.
    # `port=8000` specifies the port to listen on.
    uvicorn.run(app, host="0.0.0.0", port=8000)