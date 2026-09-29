"""Simplified jet lap environment.

This is a rigid-body toy model: thrust, lift from speed and angle of attack,
gravity, and drag. It is meant to be responsive and learnable, not a CFD model.
Controls are rate commands in [-1, 1]:

    0 throttle up/down
    1 pitch up/down
    2 roll right/left
    3 yaw right/left
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import MAX_TRACK_DISTANCE, N_GATES, build_track, distance_to_centerline

SIM_HZ = 30
DT = 1.0 / SIM_HZ
MAX_STEPS = 3000  # 100 seconds

MASS = 1.0
GRAVITY = 9.81
CRUISE_SPEED = 42.0
THRUST_MAX = 18.0
DRAG_K = 0.0042
INDUCED_K = 0.0016
LIFT_K = GRAVITY / (CRUISE_SPEED ** 2)
MAX_PITCH_RATE = 1.15
MAX_ROLL_RATE = 2.1
MAX_YAW_RATE = 0.85
RATE_TAU = 0.12
MAX_SPEED = 85.0
GROUND_Y = 1.5
CEILING_Y = 280.0
PHI_SCALE = 0.06
GATE_REWARD = 12.0
LAP_BONUS = 50.0
PAR_TIME = 52.0
CRASH_PENALTY = 40.0
STEP_PENALTY = 0.002

OBS_DIM = 22


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def body_axes(yaw: float, pitch: float, roll: float):
    """Aircraft axes in world coordinates. +roll drops the right wing."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    forward = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
    right0 = np.array([cy, 0.0, -sy], dtype=np.float64)
    up0 = np.cross(forward, right0)
    right = right0 * cr - up0 * sr
    up = up0 * cr + right0 * sr
    return forward, right, up


def _lag(current: float, target: float, dt: float, tau: float) -> float:
    alpha = 1.0 - np.exp(-dt / tau)
    return float(current + (target - current) * alpha)


class JetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.track = build_track()
        self.gate_centers = self.track["centers"]
        self.gate_normals = self.track["normals"]
        self.gate_radius = float(self.track["radius"])
        self.n_gates = N_GATES
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.max_steps = MAX_STEPS
        self._allocate()

    def _allocate(self):
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.pitch_rate = 0.0
        self.roll_rate = 0.0
        self.yaw_rate = 0.0
        self.throttle = 0.48
        self.speed = 0.0
        self.aoa = 0.0
        self.g_force = 1.0
        self.next_gate = 0
        self.cleared = np.zeros(self.n_gates, dtype=bool)
        self.prev_dist = 0.0
        self.steps = 0
        self.time = 0.0
        self.lap_start = None
        self.lap_time = None
        self.lap_running = False
        self.episode_reward = 0.0
        self.track_dist = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._allocate()
        rng = self.np_random
        self.position = self.track["start_position"] + rng.uniform(-6.0, 6.0, size=3)
        self.position[1] = float(np.clip(self.position[1], 28.0, 120.0))
        self.yaw = wrap_angle(self.track["start_yaw"] + float(rng.uniform(-0.06, 0.06)))
        self.pitch = float(self.track["start_pitch"] + rng.uniform(-0.04, 0.04))
        self.roll = float(rng.uniform(-0.04, 0.04))
        forward, right, up = body_axes(self.yaw, self.pitch, self.roll)
        launch = CRUISE_SPEED + float(rng.uniform(-2.0, 2.0))
        self.velocity = forward * launch + rng.uniform(-1.5, 1.5, size=3)
        self.throttle = float(np.clip(0.48 + rng.uniform(-0.05, 0.05), 0.0, 1.0))
        self.speed = float(np.linalg.norm(self.velocity))
        self.prev_dist = self._gate_distance()
        self.track_dist = distance_to_centerline(self.position, self.track["centerline"])
        return self._observation(), self._info(passed_gate=None, reason="")

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:4], -1.0, 1.0)
        if action.shape[0] != 4:
            raise ValueError("expected 4 controls")
        self.steps += 1
        self.time += DT

        self.throttle = float(np.clip(self.throttle + action[0] * 0.55 * DT, 0.0, 1.0))
        self.pitch_rate = _lag(self.pitch_rate, action[1] * MAX_PITCH_RATE, DT, RATE_TAU)
        self.roll_rate = _lag(self.roll_rate, action[2] * MAX_ROLL_RATE, DT, RATE_TAU)
        self.yaw_rate = _lag(self.yaw_rate, action[3] * MAX_YAW_RATE, DT, RATE_TAU)

        # Mild static stability so stick noise does not immediately tumble the nose.
        self.pitch = float(np.clip(self.pitch + (self.pitch_rate - 0.45 * self.aoa) * DT, -1.05, 1.05))
        self.roll = float(np.clip(self.roll + self.roll_rate * DT, -1.15, 1.15))
        coord_yaw = 0.0
        if self.speed > 12.0:
            bank = float(np.clip(self.roll, -1.05, 1.05))
            coord_yaw = GRAVITY * np.tan(bank) / max(self.speed, 12.0)
        self.yaw = wrap_angle(self.yaw + (self.yaw_rate + coord_yaw) * DT)

        forward, right, up = body_axes(self.yaw, self.pitch, self.roll)
        self.speed = float(np.linalg.norm(self.velocity))
        if self.speed > 2.0:
            v_hat = self.velocity / self.speed
            v_fwd = float(np.dot(self.velocity, forward))
            v_up = float(np.dot(self.velocity, up))
            self.aoa = float(np.clip(np.arctan2(-v_up, max(v_fwd, 1.0)), -0.7, 0.7))
        else:
            v_hat = forward
            self.aoa = float(self.pitch)

        cl = 1.0 + 3.6 * self.aoa
        if self.aoa > 0.42:
            cl *= 0.45
        cl = float(np.clip(cl, -0.2, 2.3))

        lift_mag = LIFT_K * (self.speed ** 2) * cl
        lift = up * lift_mag
        thrust = forward * (self.throttle * THRUST_MAX)
        drag_mag = (DRAG_K + INDUCED_K * cl * cl) * (self.speed ** 2)
        drag = -v_hat * drag_mag
        gravity = np.array([0.0, -GRAVITY, 0.0])
        accel = (thrust + lift + drag) / MASS + gravity
        # Ground effect. A hard dive still reaches the surface and ends the episode.
        if self.position[1] < 28.0:
            accel[1] += (28.0 - self.position[1]) / 28.0 * 6.5

        self.velocity = self.velocity + accel * DT
        self.speed = float(np.linalg.norm(self.velocity))
        if self.speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / self.speed
            self.speed = MAX_SPEED

        previous = self.position.copy()
        self.position = self.position + self.velocity * DT
        self.g_force = float(np.linalg.norm(lift) / GRAVITY)
        self.track_dist = distance_to_centerline(self.position, self.track["centerline"])

        reward = -STEP_PENALTY
        terminated = False
        truncated = False
        reason = ""
        passed_index = None

        below_ground = self.position[1] < GROUND_Y
        out_of_bounds = self.track_dist > MAX_TRACK_DISTANCE or self.position[1] > CEILING_Y
        invalid = not np.isfinite(self.position).all() or not np.isfinite(self.velocity).all()

        if below_ground or out_of_bounds or invalid:
            if invalid:
                self.position = np.nan_to_num(self.position)
                self.velocity = np.zeros(3)
            reward -= CRASH_PENALTY
            terminated = True
            reason = "crash_ground" if below_ground else "crash_bounds"
        else:
            passed_gate = self._check_gate(previous, self.position)
            if passed_gate is None:
                dist = self._gate_distance()
                delta = float(np.clip(self.prev_dist - dist, -8.0, 8.0))
                reward += PHI_SCALE * delta
                self.prev_dist = dist
            else:
                passed_index = int(passed_gate[0])
                offset = passed_gate[1]
                centering = 1.0 - offset / self.gate_radius
                reward += GATE_REWARD * (0.35 + 0.65 * centering)
                self._advance_gate(passed_index)
                self.prev_dist = self._gate_distance()

            if self.cleared.all():
                elapsed = max(float(self.lap_time or self.time), 1.0)
                multiplier = float(np.clip(PAR_TIME / elapsed, 0.45, 2.4))
                reward += LAP_BONUS * multiplier
                terminated = True
                reason = "lap_complete"
            elif self.steps >= self.max_steps:
                truncated = True
                reason = "timeout"

        self.episode_reward += float(reward)
        info = self._info(passed_gate=passed_index, reason=reason)
        return self._observation(), float(reward), terminated, truncated, info

    def _gate_distance(self) -> float:
        return float(np.linalg.norm(self.gate_centers[self.next_gate] - self.position))

    def _check_gate(self, previous: np.ndarray, current: np.ndarray):
        index = int(self.next_gate)
        center = self.gate_centers[index]
        normal = self.gate_normals[index]
        d0 = float(np.dot(previous - center, normal))
        d1 = float(np.dot(current - center, normal))
        if not (d0 < 0.0 <= d1):
            return None
        denom = d0 - d1
        if abs(denom) < 1e-8:
            return None
        t = d0 / denom
        if t < 0.0 or t > 1.0:
            return None
        point = previous + t * (current - previous)
        offset_vec = point - center
        offset_vec = offset_vec - normal * float(np.dot(offset_vec, normal))
        offset = float(np.linalg.norm(offset_vec))
        if offset > self.gate_radius:
            return None
        return index, offset

    def _advance_gate(self, index: int):
        self.cleared[index] = True
        if index == 0 and not self.lap_running:
            self.lap_running = True
            self.lap_start = self.time
            self.lap_time = 0.0
        if self.lap_running and self.lap_start is not None:
            self.lap_time = self.time - self.lap_start
        self.next_gate = (index + 1) % self.n_gates

    def _observation(self) -> np.ndarray:
        forward, right, up = body_axes(self.yaw, self.pitch, self.roll)
        rel = self.gate_centers[self.next_gate] - self.position
        follow = self.gate_centers[(self.next_gate + 1) % self.n_gates] - self.position
        rel_b = np.array([np.dot(rel, right), np.dot(rel, up), np.dot(rel, forward)])
        follow_b = np.array(
            [np.dot(follow, right), np.dot(follow, up), np.dot(follow, forward)]
        )
        vel_b = np.array(
            [
                np.dot(self.velocity, right),
                np.dot(self.velocity, up),
                np.dot(self.velocity, forward),
            ]
        )
        down = np.array([0.0, -1.0, 0.0])
        down_b = np.array([np.dot(down, right), np.dot(down, up), np.dot(down, forward)])
        dist = float(np.linalg.norm(rel))
        obs = np.array(
            [
                rel_b[0] / 200.0,
                rel_b[1] / 200.0,
                rel_b[2] / 200.0,
                follow_b[0] / 250.0,
                follow_b[1] / 250.0,
                follow_b[2] / 250.0,
                vel_b[0] / 60.0,
                vel_b[1] / 60.0,
                vel_b[2] / 60.0,
                down_b[0],
                down_b[1],
                down_b[2],
                self.pitch_rate / 2.0,
                self.roll_rate / 2.0,
                self.yaw_rate / 2.0,
                self.throttle,
                self.position[1] / 120.0,
                self.speed / 70.0,
                np.clip(self.aoa, -1.0, 1.0),
                self.next_gate / self.n_gates,
                dist / 300.0,
                self.g_force / 4.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)

    def _info(self, passed_gate, reason: str) -> dict:
        return {
            "reason": reason,
            "passed_gate": passed_gate,
            "gate_index": int(self.next_gate),
            "gates_cleared": [bool(v) for v in self.cleared],
            "lap_time": None if self.lap_time is None else float(self.lap_time),
            "lap_running": bool(self.lap_running),
            "speed": float(self.speed),
            "altitude": float(self.position[1]),
            "g_force": float(self.g_force),
            "throttle": float(self.throttle),
            "distance_to_track": float(self.track_dist),
            "episode_reward": float(self.episode_reward),
            "steps": int(self.steps),
        }
