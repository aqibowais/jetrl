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
    for i in range(len(track["centers"])):
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


# ---------------------------------------------------------------------------
# Procedural tracks
# ---------------------------------------------------------------------------

MIN_TURN_RADIUS = 190.0  # at cruise speed and 60 deg bank the jet turns at ~104 m; margin for the approach


def _gates_on_line(points: np.ndarray, n_gates: int, rng, radius: float = GATE_RADIUS) -> dict:
    """Place gates at (jittered) equal arc length along a closed 3D polyline."""
    seg = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    def at(s: float) -> np.ndarray:
        s = s % total
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = min(i, len(points) - 1)
        t = (s - cum[i]) / max(seg[i], 1e-9)
        return points[i] + t * (points[(i + 1) % len(points)] - points[i])

    spacing = total / n_gates
    offsets = [i * spacing + float(rng.uniform(-0.1, 0.1)) * spacing * (i > 0) for i in range(n_gates)]
    centers, normals, yaws, pitches = [], [], [], []
    for s in offsets:
        pos = at(s)
        ahead = at(s + 36.0)
        d = ahead - pos
        n = d / max(float(np.linalg.norm(d)), 1e-6)
        centers.append(pos)
        normals.append(n)
        yaws.append(float(np.arctan2(n[0], n[2])))
        pitches.append(float(np.arctan2(n[1], np.hypot(n[0], n[2]))))
    n_line = max(96, n_gates * 12)
    line = np.stack([at(total * i / n_line) for i in range(n_line)])
    start = at(total - 60.0)
    start_dir = at(total - 24.0) - start
    start_dir /= max(float(np.linalg.norm(start_dir)), 1e-6)
    return {
        "centers": np.stack(centers),
        "normals": np.stack(normals),
        "yaws": np.asarray(yaws),
        "pitches": np.asarray(pitches),
        "radius": float(radius),
        "centerline": line,
        "lap_length": total,
        "start_position": start,
        "start_yaw": float(np.arctan2(start_dir[0], start_dir[2])),
        "start_pitch": float(np.arctan2(start_dir[1], np.hypot(start_dir[0], start_dir[2]))),
    }


# The fixed circuit's own maxima: 54 deg and 21 m between consecutive gates.
MAX_GATE_TURN_DEG = 58.0
MAX_GATE_CLIMB = 22.0


def flyable(track: dict) -> bool:
    """Consecutive gates must not turn or climb more than the jet can line up for."""
    c, n = track["centers"], track["normals"]
    for i in range(len(c)):
        j = (i + 1) % len(c)
        turn = np.degrees(np.arccos(np.clip(float(n[i] @ n[j]), -1.0, 1.0)))
        if turn > MAX_GATE_TURN_DEG or abs(c[j][1] - c[i][1]) > MAX_GATE_CLIMB:
            return False
    return True


def random_track(rng, n_gates: int | None = None) -> dict:
    """A random circuit that passes flyable(); redraws until one does."""
    for _ in range(100):
        track = _random_track_once(rng, n_gates)
        if flyable(track):
            return track
    return track


def _random_track_once(rng, n_gates: int | None = None) -> dict:
    """A smooth random closed circuit the jet can physically fly.

    Gates are ~195-225 m apart, like the fixed circuit. Closer gates leave the
    jet about three seconds to line up for the next ring, which even the
    hand-written pilot misses.

    Horizontal shape: a stretched circle with two low harmonics of radial
    noise, rejected and redrawn if any curve is tighter than MIN_TURN_RADIUS.
    Altitude rolls between ~35 and 100 m. Direction of travel is random.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    for _ in range(200):
        r0 = rng.uniform(200.0, 320.0)
        sx, sz = rng.uniform(0.75, 1.3, size=2)
        r = np.ones_like(phi)
        for k in (2, 3):
            r += rng.uniform(0.0, 0.12 / (k - 1)) * np.cos(k * phi + rng.uniform(0, 2 * np.pi))
        x = r0 * sx * r * np.cos(phi)
        z = r0 * sz * r * np.sin(phi)
        # Curvature radius of the horizontal curve.
        dx, dz = np.gradient(x), np.gradient(z)
        ddx, ddz = np.gradient(dx), np.gradient(dz)
        curv = np.abs(dx * ddz - dz * ddx) / np.maximum((dx * dx + dz * dz) ** 1.5, 1e-9)
        if 1.0 / max(curv.max(), 1e-9) >= MIN_TURN_RADIUS:
            break
    a1, a2 = rng.uniform(8.0, 22.0), rng.uniform(0.0, 10.0)
    p1, p2 = rng.uniform(0, 2 * np.pi, size=2)
    y = np.clip(58.0 + a1 * np.sin(phi + p1) + a2 * np.sin(2 * phi + p2), 35.0, 100.0)
    pts = np.stack([x, y, z], axis=1)
    if rng.uniform() < 0.5:
        pts = pts[::-1].copy()
    if not n_gates:
        length = float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))
        n_gates = int(np.clip(round(length / rng.uniform(195.0, 225.0)), 7, 12))
    return _gates_on_line(pts, int(n_gates), rng)
