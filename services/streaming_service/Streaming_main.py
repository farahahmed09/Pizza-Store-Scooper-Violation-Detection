# services/streaming_service/main.py (Updated to serve violation images)
import asyncio
import pika
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os
import sqlite3
import json

app = FastAPI()
DB_PATH = "data/violations.db"
VIOLATIONS_DIR = "data/violations"

# ... (RabbitMQ, WebSocket, and startup code is unchanged) ...
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='results_queue')
active_connections: list[WebSocket] = []
async def consume_rabbitmq():
    while True:
        method_frame, header_frame, body = channel.basic_get(queue='results_queue')
        if method_frame:
            channel.basic_ack(method_frame.delivery_tag)
            for connection in active_connections:
                await connection.send_text(body.decode('utf-8'))
        else:
            await asyncio.sleep(0.01)
@app.on_event("startup")
async def startup_event(): asyncio.create_task(consume_rabbitmq())
@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: active_connections.remove(websocket)
@app.get("/")
async def get_root():
    html_file_path = 'services/frontend/index.html'
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r') as f: return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found</h1>")


@app.get("/api/violations")
async def get_violations():
    """Fetches violation records (now with file paths) from the DB."""
    violations = []
    if not os.path.exists(DB_PATH): return JSONResponse(content=[])
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, frame_index, image_path FROM violations ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        for row in rows: violations.append(dict(row))
        return JSONResponse(content=violations)
    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}"); return JSONResponse(content={"error": str(e)}, status_code=500)

# --- NEW: Endpoint to serve the saved violation images ---
@app.get("/violations/image/{image_name}")
async def get_violation_image(image_name: str):
    """Serves a specific violation image file."""
    # Basic security check
    if ".." in image_name or image_name.startswith("/"):
        return JSONResponse(content={"error": "Invalid file path"}, status_code=400)
    
    file_path = os.path.join(VIOLATIONS_DIR, image_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "Image not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)