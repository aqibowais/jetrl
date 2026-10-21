"""Export trained checkpoints and the reward log for the static web app.

    python export_web.py

Writes web/public/data/:
    track.json       gate centers, normals, centerline, start pose
    manifest.json    exported checkpoints with Python-side evaluation metrics
    curve.json       reward log, binned
    policies/*.json  actor weights (base64 float32), one file per checkpoint
and web/src/sim/fixtures/parity.json, a Python-recorded episode the TypeScript
port of the environment and the policy must reproduce (see parity.test.ts).

The metrics in the manifest come from rolling each checkpoint out in the Python
environment, not from the browser, so they stay the ground truth.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from jet_env import JetEnv
from test_env import heuristic_action
from track import track_payload

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "public" / "data"
FIXTURE = ROOT / "web" / "src" / "sim" / "fixtures" / "parity.json"


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def actor_layers(model: PPO) -> list[dict]:
    sd = model.policy.state_dict()
    names = []
    i = 0
    while f"mlp_extractor.policy_net.{i}.weight" in sd:
        names.append(f"mlp_extractor.policy_net.{i}")
        i += 2
    names.append("action_net")
    layers = []
    for name in names:
        w = sd[f"{name}.weight"].cpu().numpy()
        b = sd[f"{name}.bias"].cpu().numpy()
        layers.append({"in": int(w.shape[1]), "out": int(w.shape[0]), "w": b64(w), "b": b64(b)})
    return layers


def evaluate(model, seeds: list[int]) -> dict:
    """`model` is a PPO policy, or a function of the env for the hand-written pilot."""
    env = JetEnv()
    laps, gates, rewards, survive = [], [], [], []
    reasons: dict[str, int] = {}
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        info = {}
        while not done:
            action = model(env) if callable(model) else model.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        reason = info.get("reason") or "timeout"
        reasons[reason] = reasons.get(reason, 0) + 1
        rewards.append(env.episode_reward)
        gates.append(int(sum(env.cleared)))
        survive.append(env.time)
        if reason == "lap_complete":
            laps.append(float(env.lap_time))
    n = len(seeds)
    return {
        "episodes": n,
        "lap_rate": len(laps) / n,
        "best_lap": min(laps) if laps else None,
        "mean_lap": float(np.mean(laps)) if laps else None,
        "mean_gates": float(np.mean(gates)),
        "mean_reward": float(np.mean(rewards)),
        "mean_airtime": float(np.mean(survive)),
        "outcomes": reasons,
    }


def pick_checkpoints(rows: list[dict], count: int) -> list[dict]:
    """Evenly spaced across the run, always keeping the first and the last."""
    if len(rows) <= count:
        return rows
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, count)).astype(int))
    return [rows[i] for i in idx]


def curve(log: Path, bins: int) -> dict:
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        return {"bins": []}
    # Cumulative env steps from episode lengths. The logged SB3 counter resets
    # whenever a run restarts, so it cannot be used as an x axis across runs.
    total = 0
    for r in rows:
        total += int(r["length"])
        r["timesteps"] = total
    size = max(1, len(rows) // bins)
    out = []
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        out.append(
            {
                "episode": int(chunk[-1]["episode"]),
                "timesteps": int(chunk[-1]["timesteps"]),
                "reward": round(float(np.mean([r["reward"] for r in chunk])), 3),
                "reward_lo": round(float(np.percentile([r["reward"] for r in chunk], 10)), 3),
                "reward_hi": round(float(np.percentile([r["reward"] for r in chunk], 90)), 3),
                "gates": round(float(np.mean([r.get("gates_passed", 0) for r in chunk])), 3),
                "laps": round(float(np.mean([r.get("reason") == "lap_complete" for r in chunk])), 3),
            }
        )
    return {"episodes": len(rows), "bins": out}


def parity_fixture(model: PPO, seed: int, steps: int) -> dict:
    env = JetEnv()
    obs, _ = env.reset(seed=seed)

    def snap() -> dict:
        return {
            "position": env.position.tolist(),
            "velocity": env.velocity.tolist(),
            "heading": env.heading,
            "gamma": env.gamma,
            "bank": env.bank,
            "pull": env.pull,
            "rudder": env.rudder,
            "pitch": env.pitch,
            "roll": env.roll,
            "yaw": env.yaw,
            "pitchRate": env.pitch_rate,
            "rollRate": env.roll_rate,
            "yawRate": env.yaw_rate,
            "throttle": env.throttle,
            "speed": env.speed,
            "aoa": env.aoa,
            "gForce": env.g_force,
            "nextGate": int(env.next_gate),
            "cleared": [bool(v) for v in env.cleared],
            "prevDist": env.prev_dist,
            "steps": env.steps,
            "time": env.time,
            "lapStart": env.lap_start,
            "lapTime": env.lap_time,
            "lapRunning": env.lap_running,
            "episodeReward": env.episode_reward,
            "trackDist": env.track_dist,
            "lastProgress": env.last_progress,
        }

    initial = snap()
    frames = []
    for _ in range(steps):
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        frame = {"obs": [float(v) for v in obs], "action": [float(v) for v in action]}
        obs, reward, terminated, truncated, info = env.step(action)
        frame["reward"] = reward
        frame["state"] = snap()
        frames.append(frame)
        if terminated or truncated:
            break
    return {"policy": "fixture", "seed": seed, "initial": initial, "frames": frames, "layers": actor_layers(model)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--log", type=Path, default=ROOT / "logs" / "rewards.jsonl")
    parser.add_argument("--count", type=int, default=12, help="checkpoints to export")
    parser.add_argument("--eval-seeds", type=int, default=30)
    parser.add_argument("--generalist", type=Path, default=ROOT / "runs" / "generalist" / "checkpoints" / "ep_05000.zip")
    parser.add_argument("--generalist-test", type=Path, default=ROOT / "runs" / "generalist" / "test_ep05000.json",
                        help="eval_unseen.py output for the generalist on the held-out test tracks")
    parser.add_argument("--specialist-test", type=Path, default=ROOT / "runs" / "specialist_test.json")
    parser.add_argument("--ablation", type=Path, default=ROOT / "runs" / "ablation.json")
    parser.add_argument("--story-seed", type=int, default=2,
                        help="browser track (web/src/sim/tracks.ts) where the specialist fails and the generalist laps")
    args = parser.parse_args()

    (OUT / "policies").mkdir(parents=True, exist_ok=True)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "track.json").write_text(json.dumps(track_payload()), encoding="utf-8")

    rows = []
    for meta in sorted(args.checkpoints.glob("ep_*.json")):
        row = json.loads(meta.read_text(encoding="utf-8"))
        if (args.checkpoints / f"{row['id']}.zip").exists():
            rows.append(row)
    rows.sort(key=lambda r: int(r["episode"]))
    if not rows:
        raise SystemExit(f"no checkpoints in {args.checkpoints}")

    seeds = list(range(1000, 1000 + args.eval_seeds))
    exported = []
    best = None
    for row in pick_checkpoints(rows, args.count):
        model = PPO.load(str(args.checkpoints / f"{row['id']}.zip"), device="cpu")
        stats = evaluate(model, seeds)
        (OUT / "policies" / f"{row['id']}.json").write_text(
            json.dumps({"id": row["id"], "layers": actor_layers(model)}), encoding="utf-8"
        )
        entry = {
            "id": row["id"],
            "episode": int(row["episode"]),
            "timesteps": int(row["timesteps"]),
            "train_mean_reward": row.get("mean_reward"),
            "eval": stats,
        }
        exported.append(entry)
        score = (stats["lap_rate"], -(stats["mean_lap"] or 1e9), stats["mean_reward"])
        if best is None or score > best[0]:
            best = (score, entry["id"], model)
        print(
            f"{row['id']}  laps {stats['lap_rate']:.0%}  mean lap {stats['mean_lap'] or float('nan'):6.2f}s  "
            f"gates {stats['mean_gates']:.1f}  reward {stats['mean_reward']:8.2f}"
        )

    manifest = {
        "best": best[1],
        "eval_seeds": len(seeds),
        "checkpoints": exported,
        "story_seed": args.story_seed,
        "heuristic": evaluate(heuristic_action, seeds),
    }

    # Generalist: weights for the browser, results from the held-out test tracks
    # (eval_unseen.py --track-seed 888000), never the tracks used to pick it.
    if args.generalist.exists() and args.generalist_test.exists():
        gen = PPO.load(str(args.generalist), device="cpu")
        gid = "gen_" + args.generalist.stem.replace("ep_", "")
        (OUT / "policies" / f"{gid}.json").write_text(
            json.dumps({"id": gid, "layers": actor_layers(gen), "mask": True}), encoding="utf-8"
        )
        test = json.loads(args.generalist_test.read_text(encoding="utf-8"))[0]
        meta = json.loads(args.generalist.with_suffix(".json").read_text(encoding="utf-8"))
        manifest["generalist"] = {
            "id": gid,
            "episode": int(meta["episode"]),
            "timesteps": int(meta["timesteps"]),
            "fixed": {**test["fixed"], "episodes": test["fixed"]["tracks"], "mean_gates": test["fixed"]["gate_share"] * 9,
                      "best_lap": None, "mean_reward": None, "mean_airtime": None},
            "unseen": test["unseen"],
        }
        exported_ids = {e["id"] for e in exported} | {gid}
    else:
        exported_ids = {e["id"] for e in exported}
    if args.specialist_test.exists():
        manifest["specialist_unseen"] = json.loads(args.specialist_test.read_text(encoding="utf-8"))[0]["unseen"]
    if args.ablation.exists():
        manifest["ablation"] = json.loads(args.ablation.read_text(encoding="utf-8"))

    # Drop weights from earlier exports so the site never ships a policy the manifest doesn't describe.
    keep = {f"{i}.json" for i in exported_ids}
    for stale in (OUT / "policies").glob("*.json"):
        if stale.name not in keep:
            stale.unlink()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.log.exists():
        (OUT / "curve.json").write_text(json.dumps(curve(args.log, 160)), encoding="utf-8")
    FIXTURE.write_text(json.dumps(parity_fixture(best[2], seed=4242, steps=900)), encoding="utf-8")
    print(f"best {best[1]}; wrote {OUT}")


if __name__ == "__main__":
    main()
