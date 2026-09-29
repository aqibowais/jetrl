"""Plot the append-only reward log.

    python plot_rewards.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "rewards.jsonl"
OUT = ROOT / "logs" / "reward_curve.png"


def load_rows():
    if not LOG.exists():
        raise SystemExit(f"No reward log at {LOG}. Train first.")
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit("Reward log is empty.")
    return rows


def rolling_mean(values, window):
    means = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        means.append(total / min(i + 1, window))
    return means


def main():
    rows = load_rows()
    episodes = [row["episode"] for row in rows]
    rewards = [row["reward"] for row in rows]
    means = rolling_mean(rewards, 20)
    window = min(20, len(rewards))
    first = sum(rewards[:window]) / window
    last = sum(rewards[-window:]) / window
    print(f"episodes: {len(rewards)}")
    print(f"first {window} mean: {first:.2f}")
    print(f"last {window} mean:  {last:.2f}")
    print(f"change: {last - first:+.2f}")

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=140)
    fig.patch.set_facecolor("#0c0f0d")
    ax.set_facecolor("#0c0f0d")
    ax.plot(episodes, rewards, color="#1f6b45", linewidth=0.8, label="episode reward")
    ax.plot(episodes, means, color="#7CFF6B", linewidth=1.6, label="mean of 20")
    marks_x = []
    marks_y = []
    for row, reward in zip(rows, rewards):
        if row.get("checkpoint"):
            marks_x.append(row["episode"])
            marks_y.append(reward)
    if marks_x:
        ax.scatter(marks_x, marks_y, s=28, color="#7FE7FF", zorder=3, label="checkpoint")
    ax.set_xlabel("episode", color="#cfe8dc")
    ax.set_ylabel("reward", color="#cfe8dc")
    ax.tick_params(colors="#cfe8dc")
    for spine in ax.spines.values():
        spine.set_color("#1d3a2e")
    ax.legend(facecolor="#0c0f0d", edgecolor="#1d3a2e", labelcolor="#cfe8dc")
    ax.set_title("JetRL training reward", color="#e8f6f2", loc="left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
