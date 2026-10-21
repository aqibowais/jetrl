"""Evaluate checkpoints on procedural circuits no policy trained on.

    python eval_unseen.py runs/generalist/checkpoints --mask
    python eval_unseen.py checkpoints --only ep_09500

The tracks come from a fixed seed that training never used, so every policy
faces the same circuits.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

UNSEEN_SEED = 777_000


def unseen_tracks(n: int, seed: int = UNSEEN_SEED):
    from track import random_track

    rng = np.random.default_rng(seed)
    return [random_track(rng) for _ in range(n)]


def evaluate(path: str, mask: bool, n_tracks: int, fixed_episodes: int = 0, track_seed: int = UNSEEN_SEED) -> dict:
    from stable_baselines3 import PPO

    from jet_env import JetEnv

    model = PPO.load(path, device="cpu")
    env = JetEnv(mask_gate_index=mask)

    def roll(options: dict | None, seed: int):
        obs, _ = env.reset(seed=seed, options=options)
        done = False
        info: dict = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, te, tu, info = env.step(a)
            done = te or tu
        return info.get("reason") or "timeout", float(env.cleared.sum()) / env.n_gates, env.lap_time

    def summary(rows) -> dict:
        laps = [t for r, _, t in rows if r == "lap_complete"]
        outcomes: dict[str, int] = {}
        for r, _, _ in rows:
            outcomes[r] = outcomes.get(r, 0) + 1
        return {
            "tracks": len(rows),
            "lap_rate": len(laps) / len(rows),
            "mean_lap": float(np.mean(laps)) if laps else None,
            "gate_share": float(np.mean([g for _, g, _ in rows])),
            "outcomes": outcomes,
        }

    out = {"path": path, "unseen": summary([roll({"track": t}, 5000 + i) for i, t in enumerate(unseen_tracks(n_tracks, track_seed))])}
    if fixed_episodes:
        out["fixed"] = summary([roll(None, 1000 + i) for i in range(fixed_episodes)])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--mask", action="store_true", help="policy was trained with the gate index hidden")
    p.add_argument("--tracks", type=int, default=30)
    p.add_argument("--fixed", type=int, default=0, help="also evaluate on the fixed circuit")
    p.add_argument("--only", default=None, help="evaluate one checkpoint id")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--track-seed", type=int, default=UNSEEN_SEED, help="777000 = validation set, 888000 = test set")
    args = p.parse_args()
    paths = sorted(str(x) for x in args.dir.glob("ep_*.zip"))
    if args.only:
        paths = [x for x in paths if args.only in x]
    with Pool(args.workers) as pool:
        results = pool.starmap(evaluate, [(x, args.mask, args.tracks, args.fixed, args.track_seed) for x in paths])
    for r in results:
        u = r["unseen"]
        extra = f"  fixed laps {r['fixed']['lap_rate']:.0%}" if "fixed" in r else ""
        print(f"{Path(r['path']).stem}  unseen laps {u['lap_rate']:.0%}  gates {u['gate_share']:.0%}  mean lap {u['mean_lap'] or float('nan'):.1f}{extra}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
