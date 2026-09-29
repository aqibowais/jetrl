"""Closed lap of ring gates for JetRL.

The centerline is a stadium: two straights and two semicircles. Gates sit on
that line at equal arc length, with altitude and heading changing around the lap.
"""

from __future__ import annotations

import numpy as np

N_GATES = 9
STRAIGHT = 240.0
RADIUS = 220.0
GATE_RADIUS = 26.0
MAX_TRACK_DISTANCE = 420.0


def _altitude(distance_along: float, total: float) -> float:
    phase = 2.0 * np.pi * distance_along / total
    alt = 50.0 + 20.0 * np.sin(phase) + 10.0 * np.sin(2.0 * phase + 0.5)
    return float(np.clip(alt, 34.0, 100.0))


def _horizontal_pose(distance_along: float, straight: float, radius: float):
    """Return x, z, and a unit horizontal tangent at arc length s."""
    length = straight
    arc = np.pi * radius
    total = 2.0 * length + 2.0 * arc
    s = distance_along % total

    if s < length:
        x = -radius
        z = -length / 2.0 + s
        return x, z, 0.0, 1.0

    s -= length
    if s < arc:
        ang = np.pi - s / radius
        x = np.cos(ang) * radius
        z = length / 2.0 + np.sin(ang) * radius
        tx, tz = np.sin(ang), -np.cos(ang)
        norm = float(np.hypot(tx, tz))
        return float(x), float(z), tx / norm, tz / norm

    s -= arc
    if s < length:
        x = radius
        z = length / 2.0 - s
        return x, z, 0.0, -1.0

    s -= length
    ang = -s / radius
    x = np.cos(ang) * radius
    z = -length / 2.0 + np.sin(ang) * radius
    tx, tz = np.sin(ang), -np.cos(ang)
    norm = float(np.hypot(tx, tz))
    return float(x), float(z), tx / norm, tz / norm


def _sample(distance_along: float, total: float):
    x, z, tx, tz = _horizontal_pose(distance_along, STRAIGHT, RADIUS)
    alt = _altitude(distance_along, total)
    return np.array([x, alt, z], dtype=np.float64), np.array([tx, tz], dtype=np.float64)


def build_track(samples_per_gate: int = 12):
    """Return gate and centerline arrays used by the environment and the UI."""
    arc = np.pi * RADIUS
    total = 2.0 * STRAIGHT + 2.0 * arc
    centers = []
    normals = []
    yaws = []
    pitches = []

    for i in range(N_GATES):
        s = total * i / N_GATES
        pos, tangent = _sample(s, total)
        ahead, _ = _sample(s + 36.0, total)
        direction = ahead - pos
        norm = float(np.linalg.norm(direction))
        normal = direction / max(norm, 1e-6)
        yaw = float(np.arctan2(tangent[0], tangent[1]))
        pitch = float(np.arctan2(normal[1], np.hypot(normal[0], normal[2])))
        centers.append(pos)
        normals.append(normal)
        yaws.append(yaw)
        pitches.append(pitch)

    n_line = N_GATES * samples_per_gate
    centerline = np.zeros((n_line, 3), dtype=np.float64)
    for i in range(n_line):
        pos, _ = _sample(total * i / n_line, total)
        centerline[i] = pos

    start_pos, _ = _sample(total - 60.0, total)
    start_ahead, _ = _sample(total - 24.0, total)
    start_dir = start_ahead - start_pos
    start_dir = start_dir / max(float(np.linalg.norm(start_dir)), 1e-6)
    start_yaw = float(np.arctan2(start_dir[0], start_dir[2]))
    start_pitch = float(np.arctan2(start_dir[1], np.hypot(start_dir[0], start_dir[2])))

    return {
        "centers": np.stack(centers).astype(np.float64),
        "normals": np.stack(normals).astype(np.float64),
        "yaws": np.asarray(yaws, dtype=np.float64),
        "pitches": np.asarray(pitches, dtype=np.float64),
        "radius": GATE_RADIUS,
        "centerline": centerline,
        "lap_length": float(total),
        "start_position": start_pos.astype(np.float64),
        "start_yaw": start_yaw,
        "start_pitch": start_pitch,
    }


def track_payload(track: dict | None = None) -> dict:
    track = track or build_track()
    gates = []
    for i in range(N_GATES):
        gates.append(
            {
                "index": i,
                "center": track["centers"][i].tolist(),
                "normal": track["normals"][i].tolist(),
                "yaw": float(track["yaws"][i]),
                "pitch": float(track["pitches"][i]),
                "radius": float(track["radius"]),
            }
        )
    return {
        "gates": gates,
        "centerline": track["centerline"].tolist(),
        "lap_length": track["lap_length"],
        "start": {
            "position": track["start_position"].tolist(),
            "yaw": track["start_yaw"],
            "pitch": track["start_pitch"],
            "roll": 0.0,
            "throttle": 0.48,
        },
    }


def distance_to_centerline(point: np.ndarray, centerline: np.ndarray) -> float:
    """Shortest distance from a point to the closed centerline polyline."""
    a = centerline
    b = np.roll(centerline, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.maximum(denom, 1e-8)
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(closest - point, axis=1)))
