"""Ablation table: best checkpoint of each run within the same step budget,
evaluated on 30 starts of the fixed circuit. Writes runs/ablation.json.

    python eval_ablation.py
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUDGET = 4_000_000
EPISODES = 30

# 16 environments x 512 steps per update, same PPO settings, 4M-step budget.
# Seed 7 of the full configuration is the specialist run, scored up to 4M.
CONFIGS = [
    ("full", "Stall limit + random-gate starts", "the specialist's setup", [
        (7, ROOT / "checkpoints"),
        (21, ROOT / "runs" / "abl_full_s21" / "checkpoints"),
        (33, ROOT / "runs" / "abl_full_s33" / "checkpoints"),
    ]),
    ("nostall", "No stall limit", "no penalty for 20 s without a gate", [(7, ROOT / "runs" / "abl_nostall" / "checkpoints")]),
    ("nostart", "No random-gate starts", "every episode starts at the start line", [(7, ROOT / "runs" / "abl_nostart" / "checkpoints")]),
]


def score(path: str) -> dict:
    from stable_baselines3 import PPO

    from jet_env import JetEnv

    model = PPO.load(path, device="cpu")
    env = JetEnv()
    laps, gates = [], []
    for i in range(EPISODES):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        info: dict = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, te, tu, info = env.step(a)
            done = te or tu
        gates.append(float(env.cleared.sum()) / env.n_gates)
        if info.get("reason") == "lap_complete":
            laps.append(float(env.lap_time))
    return {"path": path, "lap_rate": len(laps) / EPISODES, "mean_lap": float(np.mean(laps)) if laps else None, "gate_share": float(np.mean(gates))}


def checkpoints(d: Path) -> list[tuple[str, int]]:
    out = []
    for meta in sorted(d.glob("ep_*.json")):
        row = json.loads(meta.read_text(encoding="utf-8"))
        z = d / f"{row['id']}.zip"
        if z.exists() and int(row["timesteps"]) <= BUDGET:
            out.append((str(z), int(row["timesteps"])))
    return out


def main() -> None:
    jobs = []
    for cid, _, _, runs in CONFIGS:
        for seed, d in runs:
            for path, steps in checkpoints(d):
                jobs.append((cid, seed, path, steps))
    with Pool(12) as pool:
        results = pool.map(score, [j[2] for j in jobs])
    table = []
    for cid, label, detail, runs in CONFIGS:
        rows = []
        for seed, _ in runs:
            mine = [(r, j) for r, j in zip(results, jobs) if j[0] == cid and j[1] == seed]
            if not mine:
                continue
            best, job = max(mine, key=lambda t: (t[0]["lap_rate"], t[0]["gate_share"]))
            rows.append({"seed": seed, "timesteps": job[3], "lap_rate": best["lap_rate"], "mean_lap": best["mean_lap"], "gate_share": best["gate_share"]})
            print(f"{cid:8s} seed {seed:3d}  best {Path(best['path']).stem}  laps {best['lap_rate']:.0%}  gates {best['gate_share']:.0%}", flush=True)
        table.append({"id": cid, "label": label, "detail": detail, "runs": rows})
    (ROOT / "runs" / "ablation.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
