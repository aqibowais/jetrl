"""Phase 1 check: random episodes, crash termination, and lap termination.

    python test_env.py
"""

from __future__ import annotations

import numpy as np

from jet_env import JetEnv, body_axes, wrap_angle


def _format_result(index, reward, steps, terminated, truncated, reason):
    end = "terminated" if terminated else "truncated" if truncated else "running"
    print(
        f"episode {index:02d}  reward {reward:8.2f}  steps {steps:4d}  {end:11s}  {reason}"
    )


def run_random(episodes: int = 4):
    env = JetEnv()
    print("random policy")
    for episode in range(1, episodes + 1):
        env.reset(seed=episode)
        total = 0.0
        terminated = False
        truncated = False
        reason = ""
        steps = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            reason = info["reason"]
            steps = info["steps"]
        _format_result(episode, total, steps, terminated, truncated, reason)
        if not (terminated or truncated):
            raise SystemExit("random episode did not end")
    env.close()


def check_axes():
    forward, right, up = body_axes(0.0, 0.0, 0.0)
    assert np.allclose(forward, [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(right, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(up, [0.0, 1.0, 0.0], atol=1e-6)
    _, right_bank, up_bank = body_axes(0.0, 0.0, 0.3)
    assert right_bank[1] < 0.0, "positive roll should drop the right wing"
    assert up_bank[0] > 0.0, "positive roll should tilt lift to the right"
    print("body axes: ok")


def check_trim():
    env = JetEnv()
    env.reset(seed=0)
    env.position[:] = [0.0, 70.0, 0.0]
    env.velocity[:] = [0.0, 0.0, 42.0]
    env.yaw = 0.0
    env.pitch = 0.0
    env.roll = 0.0
    env.pitch_rate = 0.0
    env.roll_rate = 0.0
    env.yaw_rate = 0.0
    env.throttle = 0.58
    env.speed = 42.0
    env.prev_dist = env._gate_distance()
    terminated = False
    for _ in range(90):
        _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
        if terminated or truncated:
            break
    print(f"trim altitude after 3s: {env.position[1]:.1f} m  speed {env.speed:.1f}")
    if env.position[1] < 40.0:
        raise SystemExit("trimmed flight fell out of the sky; lift sign or magnitude is wrong")
    print("trim flight: ok")
    env.close()


def check_crash():
    env = JetEnv()
    env.reset(seed=1)
    env.position[1] = 8.0
    env.velocity[:] = [0.0, -55.0, 8.0]
    reward = 0.0
    terminated = False
    truncated = False
    info = {"reason": ""}
    for _ in range(45):
        _, step_reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        reward += step_reward
        if terminated or truncated:
            break
    print(f"crash probe  reward {reward:8.2f}  reason {info['reason']}")
    if not terminated or info["reason"] != "crash_ground":
        raise SystemExit("expected crash_ground termination")
    if reward > -20.0:
        raise SystemExit("crash penalty was too small")
    print("crash termination: ok")
    env.close()


def _place_behind(env: JetEnv, index: int, back: float = 0.8, speed: float = 55.0):
    normal = env.gate_normals[index]
    center = env.gate_centers[index]
    env.position = center - normal * back
    env.velocity = normal * speed
    env.speed = speed
    env.next_gate = index
    env.prev_dist = env._gate_distance()


def check_wrong_and_missed_gate():
    env = JetEnv()
    env.reset(seed=2)
    wrong = 4
    _place_behind(env, wrong)
    env.next_gate = 0
    env.cleared[:] = False
    _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    if info["passed_gate"] is not None or env.next_gate != 0 or terminated:
        raise SystemExit("passing the wrong gate was counted")

    _place_behind(env, 0, back=0.8, speed=55.0)
    env.position = env.position + np.array([env.gate_radius + 8.0, 0.0, 0.0])
    env.next_gate = 0
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    if info["passed_gate"] is not None or env.next_gate != 0:
        raise SystemExit("a miss outside the ring was counted")
    print("wrong gate and missed gate ignored: ok")
    env.close()


def check_lap():
    env = JetEnv()
    env.reset(seed=3)
    env.cleared[:] = False
    env.lap_running = False
    env.lap_start = None
    env.lap_time = None
    total = 0.0
    reason = ""
    terminated = False
    for index in range(env.n_gates):
        _place_behind(env, index)
        passed = False
        for _ in range(6):
            _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            total += reward
            reason = info["reason"]
            if info["passed_gate"] == index:
                passed = True
                break
            if terminated or truncated:
                break
        if not passed:
            raise SystemExit(f"failed to register a centered pass of gate {index}")
    if not terminated or reason != "lap_complete":
        raise SystemExit(f"expected lap_complete, got {reason!r} terminated={terminated}")
    if total <= 0.0:
        raise SystemExit("lap completion reward was not positive")
    print(f"scripted lap  reward {total:8.2f}  lap_time {env.lap_time:.2f}s  reason {reason}")
    print("lap termination: ok")
    env.close()


def heuristic_action(env: JetEnv) -> np.ndarray:
    center = env.gate_centers[env.next_gate]
    normal = env.gate_normals[env.next_gate]
    rel = center - env.position
    dist = float(np.linalg.norm(rel))
    aim = center if dist < 50.0 else center - normal * 30.0
    offset = aim - env.position
    dist = float(np.linalg.norm(offset))
    desired = offset / max(dist, 1e-6)
    desired_yaw = float(np.arctan2(desired[0], desired[2]))
    desired_pitch = float(np.arcsin(np.clip(desired[1], -0.55, 0.55)))
    yaw_err = wrap_angle(desired_yaw - env.yaw)
    pitch_err = desired_pitch - env.pitch
    desired_roll = float(np.clip(yaw_err * 1.5, -0.95, 0.95))
    roll_err = desired_roll - env.roll
    pitch_cmd = float(np.clip(pitch_err * 2.8 - env.pitch_rate * 0.5, -1.0, 1.0))
    roll_cmd = float(np.clip(roll_err * 3.0 - env.roll_rate * 0.4, -1.0, 1.0))
    yaw_cmd = float(np.clip(yaw_err * 0.35, -1.0, 1.0))
    target_speed = 44.0 if abs(yaw_err) < 0.45 else 36.0
    if env.speed < target_speed - 1.5:
        throttle_cmd = 1.0
    elif env.speed > target_speed + 3.0:
        throttle_cmd = -0.8
    else:
        throttle_cmd = 0.1
    return np.array([throttle_cmd, pitch_cmd, roll_cmd, yaw_cmd], dtype=np.float32)


def check_heuristic():
    env = JetEnv()
    print("heuristic policy")
    completed = 0
    for seed in range(3):
        env.reset(seed=100 + seed)
        total = 0.0
        reason = ""
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(heuristic_action(env))
            total += reward
            reason = info["reason"]
        _format_result(seed + 1, total, info["steps"], terminated, truncated, reason)
        if reason == "lap_complete":
            completed += 1
            print(f"  lap time {env.lap_time:.2f}s")
    print(f"heuristic laps completed: {completed}/3")
    if completed < 1:
        raise SystemExit("heuristic could not finish a lap; the track is not flyable yet")
    env.close()


def main():
    check_axes()
    check_trim()
    run_random()
    check_crash()
    check_wrong_and_missed_gate()
    check_lap()
    check_heuristic()
    print("all environment checks passed")


if __name__ == "__main__":
    main()
