"""PPO training for the jet lap. Refuses to start without a typed confirmation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from jet_env import MAX_STEPS, JetEnv

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "logs" / "rewards.jsonl"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints"
LOCK_PATH = ROOT / "logs" / "training.lock"

POLICY_KWARGS = {"net_arch": [128, 128], "log_std_init": -1.8}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_training_lock() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if _pid_alive(pid):
            raise RuntimeError(
                "A training run is already active. Stop that run before starting another."
            )
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")


def release_training_lock() -> None:
    if not LOCK_PATH.exists():
        return
    try:
        pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        pid = -1
    if pid == os.getpid():
        LOCK_PATH.unlink(missing_ok=True)


def make_ppo(vec_env) -> PPO:
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=POLICY_KWARGS,
        device="cpu",
        verbose=0,
    )


def make_vec_env(n_envs: int, seed: int) -> DummyVecEnv:
    def factory(index):
        def _init():
            env = Monitor(JetEnv())
            env.reset(seed=seed + index)
            return env

        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


def _last_logged_episode(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = max(last, int(json.loads(line)["episode"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return last


def estimate_duration(episodes: int, n_envs: int) -> tuple[float, float]:
    env = JetEnv()
    env.reset(seed=0)
    samples = 240
    start = time.perf_counter()
    for _ in range(samples):
        env.step(env.action_space.sample())
    raw = samples / max(time.perf_counter() - start, 1e-6)
    env.close()
    # PPO spends most of its time in the optimizer, not in this numpy step.
    throughput = max(raw * 0.08 * n_envs, 1.0)
    seconds = episodes * 900 / throughput
    return seconds, raw


def format_duration(seconds: float) -> str:
    seconds = max(1, int(seconds))
    if seconds < 90:
        return f"about {seconds} seconds"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"about {minutes:.0f} minutes"
    return f"about {minutes / 60.0:.1f} hours"


def describe_plan(episodes: int, checkpoint_every: int, n_envs: int, seconds: float) -> str:
    return "\n".join(
        [
            "JETRL training plan",
            f"  episodes this run: {episodes}",
            f"  parallel environments: {n_envs}",
            f"  checkpoint every: {checkpoint_every} episodes",
            "  algorithm: PPO (Stable-Baselines3)",
            "  device: cpu",
            f"  upper bound on environment steps: {episodes * MAX_STEPS:,}",
            f"  rough duration: {format_duration(seconds)}",
            "  writes: logs/rewards.jsonl and checkpoints/",
            "This process stays running until training finishes or you stop it with Ctrl+C.",
            "Nothing has been started yet.",
        ]
    )


class TrainingLogCallback(BaseCallback):
    def __init__(
        self,
        total_episodes: int,
        checkpoint_every: int,
        log_path: Path,
        checkpoint_dir: Path,
        start_episode: int,
        on_episode=None,
        stop_event=None,
    ):
        super().__init__(verbose=0)
        self.total_episodes = total_episodes
        self.checkpoint_every = checkpoint_every
        self.log_path = log_path
        self.checkpoint_dir = checkpoint_dir
        self.next_episode = start_episode
        self.completed = 0
        self.on_episode = on_episode
        self.stop_event = stop_event
        self.recent = deque(maxlen=20)
        self._weights = None
        self.weight_version = 0

    def _on_training_start(self) -> None:
        self._publish_weights()

    def _on_step(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return False
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue
            self._record_episode(info)
            if self.completed >= self.total_episodes:
                return False
        return True

    def _record_episode(self, info: dict) -> None:
        reward = float(info["episode"]["r"])
        length = int(info["episode"]["l"])
        reason = str(info.get("reason") or "")
        lap_time = info.get("lap_time")
        if lap_time is not None:
            lap_time = float(lap_time)
        gates_passed = int(sum(bool(v) for v in (info.get("gates_cleared") or [])))
        self.recent.append(reward)
        mean_reward = float(np.mean(self.recent))
        episode = self.next_episode
        self.next_episode += 1
        self.completed += 1
        checkpoint_id = None
        if self.completed % self.checkpoint_every == 0:
            checkpoint_id = self._save_checkpoint(episode, mean_reward, reward, reason, lap_time)
        self._publish_weights()
        record = {
            "episode": episode,
            "reward": round(reward, 4),
            "length": length,
            "reason": reason,
            "lap_time": None if lap_time is None else round(lap_time, 4),
            "gates_passed": gates_passed,
            "mean_reward": round(mean_reward, 4),
            "checkpoint": checkpoint_id,
            "timesteps": int(self.num_timesteps),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        marker = f"  saved {checkpoint_id}" if checkpoint_id else ""
        print(
            f"episode {episode:4d}  reward {reward:8.2f}  mean20 {mean_reward:8.2f}  "
            f"{reason or 'running':14s}  steps {length:4d}  gates {gates_passed}{marker}",
            flush=True,
        )
        if self.on_episode is not None:
            self.on_episode(record)

    def _save_checkpoint(self, episode, mean_reward, reward, reason, lap_time) -> str:
        checkpoint_id = f"ep_{episode:05d}"
        model_path = self.checkpoint_dir / f"{checkpoint_id}.zip"
        meta_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        self.model.save(str(model_path))
        meta = {
            "id": checkpoint_id,
            "episode": episode,
            "mean_reward": round(float(mean_reward), 4),
            "episode_reward": round(float(reward), 4),
            "reason": reason,
            "lap_time": None if lap_time is None else round(float(lap_time), 4),
            "timesteps": int(self.num_timesteps),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return checkpoint_id

    def _publish_weights(self) -> None:
        weights = {key: value.detach().cpu().clone() for key, value in self.model.policy.state_dict().items()}
        self._weights = weights
        self.weight_version += 1

    def latest_weights(self):
        return self._weights, self.weight_version


def run_training(
    episodes: int,
    checkpoint_every: int,
    n_envs: int,
    seed: int,
    log_path: Path,
    checkpoint_dir: Path,
    confirmed: bool,
    on_episode=None,
    stop_event=None,
    callback_holder=None,
) -> TrainingLogCallback:
    if not confirmed:
        raise RuntimeError(
            "Training refused. An explicit confirmation is required before a run can start."
        )
    acquire_training_lock()
    try:
        return _run_training(
            episodes,
            checkpoint_every,
            n_envs,
            seed,
            log_path,
            checkpoint_dir,
            on_episode,
            stop_event,
            callback_holder,
        )
    finally:
        release_training_lock()


def _run_training(
    episodes,
    checkpoint_every,
    n_envs,
    seed,
    log_path,
    checkpoint_dir,
    on_episode,
    stop_event,
    callback_holder,
) -> TrainingLogCallback:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_episode = _last_logged_episode(log_path) + 1
    vec = make_vec_env(n_envs, seed)
    model = make_ppo(vec)
    callback = TrainingLogCallback(
        total_episodes=episodes,
        checkpoint_every=checkpoint_every,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        start_episode=start_episode,
        on_episode=on_episode,
        stop_event=stop_event,
    )
    if callback_holder is not None:
        callback_holder.append(callback)
    # Upper bound. The callback stops the run after the requested number of episodes.
    model.learn(total_timesteps=episodes * MAX_STEPS, callback=callback, progress_bar=False)
    vec.close()
    return callback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO policy to fly the JetRL lap.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.checkpoint_every < 1 or args.envs < 1:
        print("episodes, checkpoint interval, and env count must be positive.")
        return 2
    seconds, raw = estimate_duration(args.episodes, args.envs)
    print(describe_plan(args.episodes, args.checkpoint_every, args.envs, seconds))
    print(f"  raw env throughput on this machine: {raw:,.0f} steps/s (physics only, not PPO)")
    try:
        answer = input("Proceed? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Training cancelled. No run was started.")
        return 0
    print("Starting training.", flush=True)
    try:
        run_training(
            episodes=args.episodes,
            checkpoint_every=args.checkpoint_every,
            n_envs=args.envs,
            seed=args.seed,
            log_path=DEFAULT_LOG,
            checkpoint_dir=DEFAULT_CHECKPOINTS,
            confirmed=True,
        )
    except RuntimeError as exc:
        print(exc)
        return 1
    print("Training finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
