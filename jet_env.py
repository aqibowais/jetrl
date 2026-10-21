"""Jet gate-racing environment.

Point-mass flight-path model with a fly-by-wire pitch law: a centered stick
holds the flight path, pulling adds load factor, and the load factor
available is limited by speed, so a slow jet stalls and sinks. The nose sits
along the velocity vector plus the angle of attack the load factor needs.

Actions in [-1, 1]: throttle rate, pitch (load factor), roll rate, rudder.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import MAX_TRACK_DISTANCE, build_track, distance_to_centerline, random_track

SIM_HZ = 30
DT = 1.0 / SIM_HZ
MAX_STEPS = 3000  # 100 seconds

GRAVITY = 9.81
CRUISE_SPEED = 42.0
THRUST_MAX = 18.0
DRAG_K = 0.0042
INDUCED_K = 1.2          # induced drag at 1 g and cruise speed, m/s^2
AOA_1G = 0.06            # angle of attack for 1 g at cruise speed, rad
AOA_MAX = 0.32           # stall angle; caps load factor at (v / 17 m/s)^2
N_MAX = 6.0
N_MIN = -1.5
PULL_G = 3.5             # extra load factor at full back stick
PUSH_G = 1.5
MAX_ROLL_RATE = 2.2
MAX_RUDDER_RATE = 0.25
MAX_BANK = 1.35
MAX_GAMMA = 1.2
RATE_TAU = 0.12
MIN_SPEED = 8.0
MAX_SPEED = 85.0
GROUND_Y = 1.5
CEILING_Y = 280.0
PHI_SCALE = 0.06
GATE_REWARD = 12.0
LAP_BONUS = 50.0
PAR_TIME = 52.0
CRASH_PENALTY = 40.0
STEP_PENALTY = 0.002
# Ending the episode when no gate is cleared for this long removes the
# "circle safely until timeout" optimum the first runs converged to.
GATE_TIMEOUT = 20.0
STALL_PENALTY = 20.0
START_BACKOFF = 70.0

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


def attitude(forward: np.ndarray, right: np.ndarray):
    """Yaw, pitch and roll (as used by body_axes) of a body frame."""
    yaw = float(np.arctan2(forward[0], forward[2]))
    pitch = float(np.arcsin(np.clip(forward[1], -1.0, 1.0)))
    right0 = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    up0 = np.cross(forward, right0)
    roll = float(np.arctan2(-np.dot(right, up0), np.dot(right, right0)))
    return yaw, pitch, roll


def _lag(current: float, target: float, dt: float, tau: float) -> float:
    alpha = 1.0 - np.exp(-dt / tau)
    return float(current + (target - current) * alpha)


class JetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        random_start: float = 0.0,
        random_tracks: float = 0.0,
        gate_timeout: float = GATE_TIMEOUT,
        mask_gate_index: bool = False,
    ):
        """random_start: probability that reset() places the jet before a random gate.
        random_tracks: probability that reset() generates a new procedural circuit.

        Both are for training. Evaluation passes options={"track": ...} or uses
        the fixed circuit, and always starts at the start line.
        """
        super().__init__()
        self.random_start = float(random_start)
        self.random_tracks = float(random_tracks)
        self.gate_timeout = float(gate_timeout)
        # On one fixed circuit the gate index tells the policy where it is, which
        # invites memorizing the track. Generalist policies see 0 here instead.
        self.mask_gate_index = bool(mask_gate_index)
        self.fixed_track = build_track()
        self._set_track(self.fixed_track)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.max_steps = MAX_STEPS
        self._allocate()

    def _set_track(self, track: dict) -> None:
        self.track = track
        self.gate_centers = track["centers"]
        self.gate_normals = track["normals"]
        self.gate_radius = float(track["radius"])
        self.n_gates = len(track["centers"])

    def _allocate(self):
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.heading = 0.0
        self.gamma = 0.0
        self.bank = 0.0
        self.pull = 0.0
        self.rudder = 0.0
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
        self.last_progress = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        if options and options.get("track") is not None:
            self._set_track(options["track"])
        elif self.random_tracks > 0 and rng.uniform() < self.random_tracks:
            self._set_track(random_track(rng))
        else:
            self._set_track(self.fixed_track)
        self._allocate()
        self.position = self.track["start_position"] + rng.uniform(-6.0, 6.0, size=3)
        self.position[1] = float(np.clip(self.position[1], 28.0, 120.0))
        self.heading = wrap_angle(self.track["start_yaw"] + float(rng.uniform(-0.06, 0.06)))
        self.gamma = float(self.track["start_pitch"] + rng.uniform(-0.04, 0.04))
        self.bank = float(rng.uniform(-0.04, 0.04))
        self.speed = CRUISE_SPEED + float(rng.uniform(-2.0, 2.0))
        self.throttle = float(np.clip(0.48 + rng.uniform(-0.05, 0.05), 0.0, 1.0))
        start_gate = None if options is None else options.get("start_gate")
        if start_gate is None and self.random_start > 0 and rng.uniform() < self.random_start:
            start_gate = int(rng.integers(1, self.n_gates))
        if start_gate:
            self._place_before_gate(int(start_gate), rng)
        self._update_frame(1.0)
        self.prev_dist = self._gate_distance()
        self.track_dist = distance_to_centerline(self.position, self.track["centerline"])
        return self._observation(), self._info(passed_gate=None, reason="")

    def _place_before_gate(self, gate: int, rng) -> None:
        """Start on the approach to `gate`, with the earlier gates already credited.

        The lap clock is backdated by the par share of the skipped gates, so the
        lap bonus on these episodes stays on the same scale as a full lap.
        """
        center = self.gate_centers[gate]
        normal = self.gate_normals[gate]
        self.position = center - normal * START_BACKOFF + rng.uniform(-5.0, 5.0, size=3)
        self.heading = wrap_angle(float(np.arctan2(normal[0], normal[2])) + float(rng.uniform(-0.05, 0.05)))
        self.gamma = float(np.arcsin(np.clip(normal[1], -0.5, 0.5)))
        self.bank = 0.0
        self.speed = CRUISE_SPEED + float(rng.uniform(-2.0, 2.0))
        self.cleared[:gate] = True
        self.next_gate = gate
        self.lap_running = True
        self.lap_start = -PAR_TIME * gate / self.n_gates
        self.lap_time = self.time - self.lap_start

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:4], -1.0, 1.0)
        if action.shape[0] != 4:
            raise ValueError("expected 4 controls")
        self.steps += 1
        self.time += DT

        self.throttle = float(np.clip(self.throttle + action[0] * 0.55 * DT, 0.0, 1.0))
        self.pull = _lag(self.pull, float(action[1]), DT, RATE_TAU)
        self.roll_rate = _lag(self.roll_rate, float(action[2]) * MAX_ROLL_RATE, DT, RATE_TAU)
        self.rudder = _lag(self.rudder, float(action[3]) * MAX_RUDDER_RATE, DT, RATE_TAU)

        old_bank = self.bank
        self.bank = float(np.clip(self.bank + self.roll_rate * DT, -MAX_BANK, MAX_BANK))
        self.roll_rate = (self.bank - old_bank) / DT

        v = max(self.speed, MIN_SPEED)
        q = (v / CRUISE_SPEED) ** 2
        n_avail = min(AOA_MAX / AOA_1G * q, N_MAX)
        trim = np.cos(self.gamma) / np.cos(self.bank)
        n = float(np.clip(trim + self.pull * (PULL_G if self.pull > 0 else PUSH_G), N_MIN, n_avail))

        gamma_dot = GRAVITY / v * (n * np.cos(self.bank) - np.cos(self.gamma))
        heading_dot = GRAVITY / v * n * np.sin(self.bank) / np.cos(self.gamma) + self.rudder
        old_gamma = self.gamma
        self.gamma = float(np.clip(self.gamma + gamma_dot * DT, -MAX_GAMMA, MAX_GAMMA))
        self.pitch_rate = (self.gamma - old_gamma) / DT
        self.yaw_rate = float(heading_dot)
        self.heading = wrap_angle(self.heading + heading_dot * DT)

        drag = DRAG_K * v * v + INDUCED_K * n * n / q
        accel = self.throttle * THRUST_MAX - drag - GRAVITY * np.sin(self.gamma)
        self.speed = float(np.clip(self.speed + accel * DT, MIN_SPEED, MAX_SPEED))
        self.g_force = n
        self._update_frame(q)

        previous = self.position.copy()
        self.position = self.position + self.velocity * DT
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
            elif self.time - self.last_progress > self.gate_timeout:
                reward -= STALL_PENALTY
                terminated = True
                reason = "stalled"
            elif self.steps >= self.max_steps:
                truncated = True
                reason = "timeout"

        self.episode_reward += float(reward)
        info = self._info(passed_gate=passed_index, reason=reason)
        return self._observation(), float(reward), terminated, truncated, info

    def set_velocity(self, velocity) -> None:
        velocity = np.asarray(velocity, dtype=np.float64)
        self.speed = float(np.clip(np.linalg.norm(velocity), MIN_SPEED, MAX_SPEED))
        self.heading = float(np.arctan2(velocity[0], velocity[2]))
        self.gamma = float(np.clip(np.arcsin(velocity[1] / max(np.linalg.norm(velocity), 1e-6)), -MAX_GAMMA, MAX_GAMMA))
        self._update_frame((self.speed / CRUISE_SPEED) ** 2)

    def _update_frame(self, q: float) -> None:
        """Velocity from the flight path; body attitude adds the angle of attack."""
        path, right, lift_dir = body_axes(self.heading, self.gamma, self.bank)
        self.velocity = path * self.speed
        self.aoa = float(np.clip(self.g_force * AOA_1G / max(q, 1e-3), -0.15, AOA_MAX))
        ca, sa = np.cos(self.aoa), np.sin(self.aoa)
        forward = path * ca + lift_dir * sa
        self.yaw, self.pitch, self.roll = attitude(forward, right)

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
        self.last_progress = self.time
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
                0.0 if self.mask_gate_index else self.next_gate / self.n_gates,
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
