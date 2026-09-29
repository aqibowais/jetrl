"""JetRL server.

Training never starts on launch. A run begins only after an explicit
confirmation from train.py or from the cockpit's confirm control.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from stable_baselines3 import PPO

from jet_env import DT, JetEnv
from track import track_payload
from train import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_LOG,
    describe_plan,
    estimate_duration,
    make_ppo,
    make_vec_env,
    run_training,
)

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
BEST_PATH = ROOT / "logs" / "best_times.json"
HOST = "127.0.0.1"
PORT = 8000


class Session:
    def __init__(self):
        self.mode = "idle"
        self.speed = 1.0
        self.training = False
        self.train_stop = threading.Event()
        self.weights = None
        self.weight_version = 0
        self.weights_lock = threading.Lock()
        self.manual_stick = np.zeros(4, dtype=np.float64)
        self.manual_needs_reset = False
        self.replay_token = 0
        self.playback_mode = "showcase"
        self.loop_id = None
        self.showcase_id = None
        self.loop = None
        self.clients: set[WebSocket] = set()
        self.last_state = None
        self.best = {
            "manual": None,
            "ai": None,
            "ai_checkpoint": None,
            "last_manual": None,
        }
        self.detail = "Idle. Training has not been started."

    def bump_replay(self) -> int:
        self.replay_token += 1
        return self.replay_token


session = Session()
track = track_payload()


def load_best() -> None:
    if not BEST_PATH.exists():
        return
    try:
        saved = json.loads(BEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for key in ("manual", "ai", "ai_checkpoint", "last_manual"):
        if key in saved:
            session.best[key] = saved[key]


def save_best() -> None:
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PATH.write_text(json.dumps(session.best, indent=2), encoding="utf-8")


def consider_ai_lap(lap_time, checkpoint_id) -> None:
    if lap_time is None:
        return
    lap_time = float(lap_time)
    current = session.best["ai"]
    if current is None or lap_time < current:
        session.best["ai"] = round(lap_time, 4)
        session.best["ai_checkpoint"] = checkpoint_id
        save_best()


def consider_manual_lap(lap_time) -> None:
    if lap_time is None:
        return
    lap_time = float(lap_time)
    session.best["last_manual"] = round(lap_time, 4)
    current = session.best["manual"]
    if current is None or lap_time < current:
        session.best["manual"] = round(lap_time, 4)
    save_best()


def playable_checkpoints() -> list[dict]:
    rows = []
    for row in list_checkpoints():
        checkpoint_id = str(row.get("id") or "")
        if checkpoint_id and (DEFAULT_CHECKPOINTS / f"{checkpoint_id}.zip").exists():
            rows.append(row)
    return rows


def best_checkpoint_id() -> str | None:
    rows = playable_checkpoints()
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("mean_reward") or -1e9))
    return str(best["id"])


def list_checkpoints() -> list[dict]:
    if not DEFAULT_CHECKPOINTS.exists():
        return []
    rows = []
    for path in sorted(DEFAULT_CHECKPOINTS.glob("ep_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda row: int(row.get("episode", 0)))
    return rows


def load_rewards() -> list[dict]:
    if not DEFAULT_LOG.exists():
        return []
    rows = []
    for line in DEFAULT_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def state_message(env: JetEnv, info: dict, reward: float, mode: str, checkpoint=None) -> dict:
    return {
        "type": "state",
        "mode": mode,
        "time": float(env.time),
        "position": [float(v) for v in env.position],
        "velocity": [float(v) for v in env.velocity],
        "pitch": float(env.pitch),
        "roll": float(env.roll),
        "yaw": float(env.yaw),
        "throttle": float(env.throttle),
        "speed": float(env.speed),
        "altitude": float(env.position[1]),
        "g_force": float(env.g_force),
        "gate_index": int(env.next_gate),
        "passed_gate": info.get("passed_gate"),
        "cleared": info.get("gates_cleared") or [False] * env.n_gates,
        "lap_time": info.get("lap_time"),
        "lap_running": bool(info.get("lap_running")),
        "reward": float(reward),
        "episode_reward": float(env.episode_reward),
        "reason": info.get("reason") or "",
        "checkpoint": checkpoint,
        "playback": session.speed if mode == "replay" else 1,
        "best": session.best,
    }


def status_message() -> dict:
    return {
        "type": "status",
        "mode": session.mode,
        "training": session.training,
        "speed": session.speed,
        "detail": session.detail,
        "best": session.best,
    }


def hello_message() -> dict:
    return {
        "type": "hello",
        "track": track,
        "checkpoints": list_checkpoints(),
        "rewards": load_rewards(),
        "best": session.best,
        "status": status_message(),
    }


async def broadcast(message: dict) -> None:
    if message.get("type") == "state":
        session.last_state = message
    dead = []
    for client in list(session.clients):
        try:
            await client.send_json(message)
        except Exception:
            dead.append(client)
    for client in dead:
        session.clients.discard(client)


def publish_threadsafe(message: dict) -> None:
    if session.loop is None:
        return
    asyncio.run_coroutine_threadsafe(broadcast(message), session.loop)


def remember_weights(callback) -> None:
    weights, version = callback.latest_weights()
    with session.weights_lock:
        session.weights = weights
        session.weight_version = version


def peek_weights():
    with session.weights_lock:
        return session.weights, session.weight_version


def on_training_episode(record: dict) -> None:
    holder = getattr(session, "callback_holder", None)
    if holder:
        remember_weights(holder[0])
    if record.get("reason") == "lap_complete":
        consider_ai_lap(record.get("lap_time"), record.get("checkpoint"))
    message = {
        "type": "metric",
        "episode": record["episode"],
        "reward": record["reward"],
        "length": record["length"],
        "reason": record["reason"],
        "lap_time": record["lap_time"],
        "mean_reward": record["mean_reward"],
        "checkpoint": record["checkpoint"],
        "timestamp": record["timestamp"],
        "best": session.best,
    }
    publish_threadsafe(message)


def training_thread(episodes: int, checkpoint_every: int, n_envs: int) -> None:
    holder = []
    session.callback_holder = holder
    try:
        run_training(
            episodes=episodes,
            checkpoint_every=checkpoint_every,
            n_envs=n_envs,
            seed=7,
            log_path=DEFAULT_LOG,
            checkpoint_dir=DEFAULT_CHECKPOINTS,
            confirmed=True,
            on_episode=on_training_episode,
            stop_event=session.train_stop,
            callback_holder=holder,
        )
        session.detail = "Training finished."
    except Exception as exc:
        session.detail = f"Training stopped: {exc}"
        print(session.detail, flush=True)
    finally:
        session.training = False
        if session.mode == "live":
            session.mode = "idle"
        publish_threadsafe(
            {
                "type": "training_finished",
                "detail": session.detail,
                "checkpoints": list_checkpoints(),
                "best": session.best,
            }
        )
        publish_threadsafe(status_message())


def manual_action(env: JetEnv) -> np.ndarray:
    stick = np.clip(session.manual_stick, -1.0, 1.0)
    throttle = float(stick[0])
    pitch = float(stick[1])
    roll_stick = float(stick[2])
    yaw = float(stick[3])
    # Digital roll keys hold a bank angle. The stick is converted to the same
    # rate command the environment already accepts, then JetEnv steps the physics.
    target_roll = roll_stick * 0.9
    roll = float(np.clip((target_roll - env.roll) * 2.5 - env.roll_rate * 0.25, -1.0, 1.0))
    return np.array([throttle, pitch, roll, yaw], dtype=np.float32)


async def follow_reward_log() -> None:
    """Stream new reward-log rows when training was started from the CLI."""
    seen = len(load_rewards())
    while True:
        await asyncio.sleep(0.6)
        if session.training:
            seen = len(load_rewards())
            continue
        rows = load_rewards()
        if len(rows) <= seen:
            continue
        fresh = rows[seen:]
        seen = len(rows)
        for row in fresh:
            await broadcast(
                {
                    "type": "metric",
                    "episode": row.get("episode"),
                    "reward": row.get("reward"),
                    "length": row.get("length"),
                    "reason": row.get("reason") or "",
                    "lap_time": row.get("lap_time"),
                    "mean_reward": row.get("mean_reward"),
                    "checkpoint": row.get("checkpoint"),
                    "timestamp": row.get("timestamp"),
                    "gates_passed": row.get("gates_passed"),
                    "best": session.best,
                }
            )
        if any(row.get("checkpoint") for row in fresh):
            await broadcast({"type": "checkpoints", "checkpoints": list_checkpoints()})


async def run_replay(checkpoint_id: str, token: int) -> bool:
    model_path = DEFAULT_CHECKPOINTS / f"{checkpoint_id}.zip"
    model = PPO.load(str(model_path), device="cpu")
    env = JetEnv()
    obs, info = env.reset()
    await broadcast(
        {
            "type": "replay_start",
            "checkpoint": checkpoint_id,
            "speed": session.speed,
        }
    )
    await broadcast(state_message(env, info, 0.0, "replay", checkpoint_id))
    finished = False
    while session.replay_token == token and session.mode == "replay":
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        await broadcast(state_message(env, info, reward, "replay", checkpoint_id))
        if terminated or truncated:
            finished = True
            if info.get("reason") == "lap_complete":
                consider_ai_lap(env.lap_time, checkpoint_id)
            await broadcast(
                {
                    "type": "replay_done",
                    "checkpoint": checkpoint_id,
                    "reason": info.get("reason") or "",
                    "lap_time": None if env.lap_time is None else float(env.lap_time),
                    "reward": float(env.episode_reward),
                    "best": session.best,
                }
            )
            break
        await asyncio.sleep(DT / float(session.speed))
    env.close()
    return finished


async def playback_job(token: int) -> None:
    """Loop a saved policy, or walk the checkpoint list. Neither path trains."""
    try:
        while session.replay_token == token and session.mode == "replay":
            rows = playable_checkpoints()
            ids = [str(row["id"]) for row in rows]
            if session.playback_mode == "progress":
                if not ids:
                    break
                if session.loop_id not in ids:
                    session.loop_id = ids[0]
            checkpoint_id = session.loop_id
            if not checkpoint_id:
                break
            kind = "Checkpoint tour" if session.playback_mode == "progress" else "Saved policy"
            session.detail = f"{kind} {checkpoint_id} at {int(session.speed)}x."
            await broadcast(status_message())
            await run_replay(checkpoint_id, token)
            if session.replay_token != token or session.mode != "replay":
                break
            if session.playback_mode == "progress" and checkpoint_id in ids:
                session.loop_id = ids[(ids.index(checkpoint_id) + 1) % len(ids)]
    finally:
        if session.replay_token == token and session.mode == "replay":
            session.mode = "live" if session.training else "idle"
            session.detail = "Playback ended."
            await broadcast(status_message())


def start_playback(checkpoint_id: str, mode: str) -> None:
    session.playback_mode = mode
    session.loop_id = checkpoint_id
    if mode == "showcase":
        session.showcase_id = checkpoint_id
    session.mode = "replay"
    token = session.bump_replay()
    asyncio.create_task(playback_job(token))


async def compare_job(first: str, second: str, token: int) -> None:
    try:
        await broadcast({"type": "compare", "phase": "a", "checkpoint": first, "against": second})
        await run_replay(first, token)
        if session.replay_token != token:
            return
        await asyncio.sleep(0.6)
        await broadcast({"type": "compare", "phase": "b", "checkpoint": second, "against": first})
        await run_replay(second, token)
        if session.replay_token == token:
            await broadcast({"type": "compare", "phase": "done", "a": first, "b": second})
    finally:
        if session.replay_token == token and session.mode == "replay":
            showcase = session.showcase_id or best_checkpoint_id()
            if showcase and not session.training:
                session.detail = f"Saved policy {showcase}."
                start_playback(showcase, "showcase")
            else:
                session.mode = "live" if session.training else "idle"
                session.detail = "Comparison finished."
                await broadcast(status_message())


async def display_loop() -> None:
    live_env = JetEnv()
    live_obs, live_info = live_env.reset(seed=11)
    manual_env = JetEnv()
    manual_obs, manual_info = manual_env.reset(seed=13)
    model = None
    loaded_version = -1
    try:
        while True:
            if session.mode == "manual":
                if session.manual_needs_reset:
                    manual_obs, manual_info = manual_env.reset()
                    session.manual_needs_reset = False
                    await broadcast(state_message(manual_env, manual_info, 0.0, "manual"))
                action = manual_action(manual_env)
                manual_obs, reward, terminated, truncated, manual_info = manual_env.step(action)
                await broadcast(state_message(manual_env, manual_info, reward, "manual"))
                if terminated or truncated:
                    if manual_info.get("reason") == "lap_complete":
                        consider_manual_lap(manual_env.lap_time)
                        await broadcast(
                            {
                                "type": "manual_lap",
                                "lap_time": float(manual_env.lap_time or 0.0),
                                "best": session.best,
                            }
                        )
                    await asyncio.sleep(1.1)
                    if session.mode == "manual":
                        manual_obs, manual_info = manual_env.reset()
                        await broadcast(state_message(manual_env, manual_info, 0.0, "manual"))
                    continue
                await asyncio.sleep(DT)
                continue

            if session.mode == "live" and session.training:
                weights, version = peek_weights()
                if weights is not None:
                    if model is None:
                        model = make_ppo(make_vec_env(1, seed=99))
                    if version != loaded_version:
                        model.policy.load_state_dict(weights)
                        loaded_version = version
                    action, _ = model.predict(live_obs, deterministic=True)
                    live_obs, reward, terminated, truncated, live_info = live_env.step(action)
                    await broadcast(state_message(live_env, live_info, reward, "live"))
                    if terminated or truncated:
                        if live_info.get("reason") == "lap_complete":
                            consider_ai_lap(live_env.lap_time, "live")
                        live_obs, live_info = live_env.reset()
                    await asyncio.sleep(DT)
                    continue

            await asyncio.sleep(0.12)
    finally:
        live_env.close()
        manual_env.close()


def parse_speed(value) -> int:
    try:
        speed = int(value)
    except (TypeError, ValueError):
        speed = 0
    if speed not in (1, 2, 4):
        raise HTTPException(status_code=400, detail="Playback speed must be 1, 2, or 4.")
    return speed


def require_checkpoint(checkpoint_id: str) -> str:
    if not checkpoint_id or not (DEFAULT_CHECKPOINTS / f"{checkpoint_id}.zip").exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found.")
    return checkpoint_id


@asynccontextmanager
async def lifespan(_app):
    load_best()
    session.loop = asyncio.get_running_loop()
    showcase = best_checkpoint_id()
    if showcase:
        session.detail = f"Saved policy {showcase}, real time."
        session.speed = 1
        start_playback(showcase, "showcase")
    else:
        session.detail = "No saved policy on disk."
    task = asyncio.create_task(display_loop())
    tail = asyncio.create_task(follow_reward_log())
    print(f"JetRL ready at http://{HOST}:{PORT}")
    print("Training does not start with the server. A saved checkpoint plays on open.")
    yield
    session.train_stop.set()
    task.cancel()
    tail.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/track")
def api_track():
    return track


@app.get("/api/checkpoints")
def api_checkpoints():
    return list_checkpoints()


@app.get("/api/rewards")
def api_rewards():
    return load_rewards()


@app.get("/api/status")
def api_status():
    return status_message()


@app.get("/api/plan")
def api_plan(episodes: int = 200, checkpoint_every: int = 20, envs: int = 4):
    episodes = int(np.clip(episodes, 1, 5000))
    checkpoint_every = int(np.clip(checkpoint_every, 1, 500))
    envs = int(np.clip(envs, 1, 8))
    seconds, _raw = estimate_duration(episodes, envs)
    return {
        "episodes": episodes,
        "checkpoint_every": checkpoint_every,
        "envs": envs,
        "estimated_seconds": seconds,
        "plan": describe_plan(episodes, checkpoint_every, envs, seconds),
    }


@app.post("/api/train")
async def api_train(body: dict):
    if body.get("confirm") is not True:
        raise HTTPException(
            status_code=400,
            detail="Training was not started. Confirm the run explicitly first.",
        )
    if session.training:
        raise HTTPException(status_code=409, detail="A training run is already active.")
    try:
        episodes = int(body.get("episodes", 200))
        checkpoint_every = int(body.get("checkpoint_every", 20))
        n_envs = int(body.get("envs", 4))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid training settings.")
    if not (1 <= episodes <= 5000 and 1 <= checkpoint_every <= 500 and 1 <= n_envs <= 8):
        raise HTTPException(status_code=400, detail="Training settings are out of range.")
    session.train_stop.clear()
    session.training = True
    session.replay_token += 1
    session.mode = "live"
    session.detail = f"Training {episodes} episodes."
    thread = threading.Thread(
        target=training_thread,
        args=(episodes, checkpoint_every, n_envs),
        daemon=True,
        name="jetrl-train",
    )
    thread.start()
    await broadcast(status_message())
    return {"started": True, "episodes": episodes, "checkpoint_every": checkpoint_every}


@app.post("/api/train/stop")
async def api_train_stop():
    if not session.training:
        return {"stopped": False}
    session.train_stop.set()
    session.detail = "Stopping training after the current step."
    await broadcast(status_message())
    return {"stopped": True}


@app.post("/api/replay")
async def api_replay(body: dict):
    checkpoint_id = require_checkpoint(str(body.get("checkpoint", "")))
    speed = parse_speed(body.get("speed", session.speed))
    session.speed = speed
    session.detail = f"Saved policy {checkpoint_id} at {speed}x."
    start_playback(checkpoint_id, "showcase")
    await broadcast(status_message())
    return {"started": True, "checkpoint": checkpoint_id, "speed": speed}


@app.post("/api/progress")
async def api_progress(body: dict | None = None):
    rows = playable_checkpoints()
    if not rows:
        raise HTTPException(status_code=404, detail="No checkpoints to play.")
    speed = parse_speed((body or {}).get("speed", session.speed))
    session.speed = speed
    session.detail = f"Checkpoint tour at {speed}x."
    start_playback(str(rows[0]["id"]), "progress")
    await broadcast(status_message())
    return {"started": True, "speed": speed, "from": rows[0]["id"]}


@app.post("/api/compare")
async def api_compare(body: dict):
    first = require_checkpoint(str(body.get("a", "")))
    second = require_checkpoint(str(body.get("b", "")))
    speed = parse_speed(body.get("speed", session.speed))
    token = session.bump_replay()
    session.mode = "replay"
    session.speed = speed
    session.detail = f"Comparing {first} then {second} at {speed}x."
    asyncio.create_task(compare_job(first, second, token))
    await broadcast(status_message())
    return {"started": True, "a": first, "b": second, "speed": speed}


@app.post("/api/speed")
async def api_speed(body: dict):
    session.speed = parse_speed(body.get("speed", 1))
    await broadcast(status_message())
    return {"speed": session.speed}


@app.post("/api/fly")
async def api_fly(body: dict):
    enabled = bool(body.get("enabled"))
    if enabled:
        session.bump_replay()
        session.mode = "manual"
        session.manual_stick[:] = 0.0
        session.manual_needs_reset = True
        session.detail = "Manual flight. W/S throttle, arrows pitch and roll, A/D yaw."
    elif session.mode == "manual":
        showcase = session.showcase_id or best_checkpoint_id()
        if showcase and not session.training:
            session.detail = f"Saved policy {showcase}."
            start_playback(showcase, "showcase")
        else:
            session.mode = "live" if session.training else "idle"
            session.detail = "Manual flight ended."
    await broadcast(status_message())
    return {"mode": session.mode}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session.clients.add(websocket)
    try:
        await websocket.send_json(hello_message())
        if session.last_state is not None:
            stamped = dict(session.last_state)
            stamped["mode"] = session.mode
            await websocket.send_json(stamped)
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                continue
            if data.get("type") == "control" and session.mode == "manual":
                session.manual_stick[:] = [
                    float(data.get("throttle", 0.0)),
                    float(data.get("pitch", 0.0)),
                    float(data.get("roll", 0.0)),
                    float(data.get("yaw", 0.0)),
                ]
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        session.clients.discard(websocket)
        if not session.clients and session.mode == "manual":
            session.mode = "live" if session.training else "idle"


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
