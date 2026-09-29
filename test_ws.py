"""Minimal client for the JetRL websocket.

Connects, rejects an unconfirmed training request, and prints a short
stream of state messages from manual flight or from a checkpoint replay.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request

import websockets

BASE = "http://127.0.0.1:8000"


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, body


async def main():
    status, body = post("/api/train", {"confirm": False, "episodes": 4})
    print(f"unconfirmed train -> {status} {body}")
    if status != 400:
        raise SystemExit("training started without confirmation")

    checkpoints = json.loads(urllib.request.urlopen(BASE + "/api/checkpoints").read().decode("utf-8"))
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        hello = json.loads(await ws.recv())
        print(f"hello type={hello.get('type')} gates={len(hello.get('track', {}).get('gates', []))}")
        if checkpoints:
            checkpoint = checkpoints[-1]["id"]
            status, body = post("/api/replay", {"checkpoint": checkpoint, "speed": 4})
            print(f"replay {checkpoint} -> {status}")
        else:
            status, body = post("/api/fly", {"enabled": True})
            print(f"manual -> {status} {body}")
            await ws.send(json.dumps({"type": "control", "throttle": 1, "pitch": 0, "roll": 0, "yaw": 0}))

        states = 0
        metrics = 0
        for _ in range(40):
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "state":
                states += 1
                if states <= 3 or states % 10 == 0:
                    pos = message["position"]
                    print(
                        f"state mode={message['mode']} "
                        f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
                        f"spd={message['speed']:.1f} gate={message['gate_index']}"
                    )
            elif kind == "metric":
                metrics += 1
                print(f"metric episode={message['episode']} reward={message['reward']}")
            if states >= 8:
                break
        print(f"received states={states} metrics={metrics}")
        if states < 3:
            raise SystemExit("expected a stream of state messages")


if __name__ == "__main__":
    asyncio.run(main())
