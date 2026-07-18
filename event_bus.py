"""
Event Bus — bridges the real IndustrialMind pipeline to the live browser demo.

Run this FIRST, before demo_live.py:
    python event_bus.py
Then open index.html in a browser (it connects to ws://localhost:8000/ws).
Then in a second terminal:
    python demo_live.py

Every event emitted by demo_live.py (via `emit()` in pipeline_events.py) is
POSTed here and immediately broadcast to every connected browser. No fake
timing, no scripted animation — the browser shows exactly what the pipeline
is doing, when it's doing it.
"""

import asyncio
import json
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_connections: List[WebSocket] = []
_event_log: List[dict] = []  # replay buffer, so late-connecting browsers still see prior events


class EventIn(BaseModel):
    type: str
    payload: dict = {}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connections.append(websocket)
    # Replay everything that already happened, in order, so the browser can
    # be opened before OR after demo_live.py starts.
    for evt in _event_log:
        await websocket.send_text(json.dumps(evt))
    try:
        while True:
            # We don't expect messages from the browser, but keep the socket
            # open and drain anything it sends (e.g. browser ping/pong).
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections.remove(websocket)


@app.post("/emit")
async def emit(event: EventIn):
    record = {"type": event.type, "payload": event.payload}
    _event_log.append(record)
    dead = []
    for ws in _connections:
        try:
            await ws.send_text(json.dumps(record))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _connections:
            _connections.remove(ws)
    return {"status": "ok", "delivered_to": len(_connections)}


@app.post("/reset")
async def reset():
    """Clear the replay buffer — call this before a fresh demo run."""
    _event_log.clear()
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok", "connections": len(_connections), "events_logged": len(_event_log)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
