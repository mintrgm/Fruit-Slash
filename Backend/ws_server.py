import asyncio
import json
import threading
import websockets

_clients = set()
_loop = None
reset_requested = False

async def _handler(websocket):
    _clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("action") == "reset":
                global reset_requested
                reset_requested = True
    finally:
        _clients.discard(websocket)

async def _broadcast_raw(payload):
    if _clients:
        await asyncio.gather(*(c.send(payload) for c in _clients), return_exceptions=True)

def broadcast_frame(frame_b64, score, lives, game_over):
    if _loop is not None:
        payload = json.dumps({
            "type": "frame",
            "data": frame_b64,
            "score": score,
            "lives": lives,
            "gameOver": game_over,
        })
        asyncio.run_coroutine_threadsafe(_broadcast_raw(payload), _loop)

def _run_server():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def main():
        async with websockets.serve(_handler, "localhost", 8765):
            await asyncio.Future()

    _loop.run_until_complete(main())

def start_server():
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()