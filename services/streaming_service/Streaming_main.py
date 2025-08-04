# services/streaming_service/main.py (Updated to handle JSON)
import asyncio
import pika
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import os

app = FastAPI()
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='results_queue')

active_connections: list[WebSocket] = []

async def consume_rabbitmq():
    """Consumes JSON messages from RabbitMQ and broadcasts them as text."""
    while True:
        method_frame, header_frame, body = channel.basic_get(queue='results_queue')
        if method_frame:
            channel.basic_ack(method_frame.delivery_tag)
            # Broadcast the JSON message (as a text string)
            for connection in active_connections:
                await connection.send_text(body.decode('utf-8'))
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
    html_file_path = 'services/frontend/index.html'
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found</h1>")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)