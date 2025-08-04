import asyncio
import pika
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import os

# --- FastAPI App Initialization ---
app = FastAPI()

# --- RabbitMQ Connection Setup ---
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
# Ensure the queue exists
channel.queue_declare(queue='results_queue')

# --- WebSocket Management ---
active_connections: list[WebSocket] = []

async def consume_rabbitmq():
    """Consumes messages from RabbitMQ and broadcasts them to all WebSocket clients."""
    while True:
        method_frame, header_frame, body = channel.basic_get(queue='results_queue')
        if method_frame:
            channel.basic_ack(method_frame.delivery_tag)
            # Broadcast the frame to all connected clients
            for connection in active_connections:
                await connection.send_bytes(body)
        else:
            await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(consume_rabbitmq())

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("A frontend client disconnected")

@app.get("/")
async def get():
    """Serves the main HTML page."""
    # Construct the path to the index.html file
    # This assumes the script is run from the project root. A more robust solution would use absolute paths.
    html_file_path = 'services/frontend/index.html'
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found</h1>")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)