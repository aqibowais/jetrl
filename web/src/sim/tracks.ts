import type { Track, Vec3 } from "./jet";

/**
 * Port of track.random_track(): same shape distribution as the circuits the
 * generalist trained on, generated in the browser so every visitor can make
 * a track no policy has seen.
 */

const GATE_RADIUS = 26;
const MIN_TURN_RADIUS = 190;
const MAX_GATE_TURN_DEG = 58;
const MAX_GATE_CLIMB = 22;

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

function gatesOnLine(points: Vec3[], nGates: number, rand: () => number): Track {
  const n = points.length;
  const seg = points.map((p, i) => norm(sub(points[(i + 1) % n], p)));
  const cum = [0];
  for (const s of seg) cum.push(cum[cum.length - 1] + s);
  const total = cum[n];
  const at = (s0: number): Vec3 => {
    const s = ((s0 % total) + total) % total;
    let i = 0;
    while (i < n - 1 && cum[i + 1] <= s) i += 1;
    const t = (s - cum[i]) / Math.max(seg[i], 1e-9);
    const a = points[i];
    const b = points[(i + 1) % n];
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
  };
  const spacing = total / nGates;
  const centers: Vec3[] = [];
  const normals: Vec3[] = [];
  for (let i = 0; i < nGates; i += 1) {
    const s = i * spacing + (i > 0 ? (rand() * 0.2 - 0.1) * spacing : 0);
    const pos = at(s);
    const d = sub(at(s + 36), pos);
    const l = Math.max(norm(d), 1e-6);
    centers.push(pos);
    normals.push([d[0] / l, d[1] / l, d[2] / l]);
  }
  const nLine = Math.max(96, nGates * 12);
  const centerline = Array.from({ length: nLine }, (_, i) => at((total * i) / nLine));
  const start = at(total - 60);
  const dir = sub(at(total - 24), start);
  return {
    centers,
    normals,
    radius: GATE_RADIUS,
    centerline,
    lapLength: total,
    start: {
      position: start,
      yaw: Math.atan2(dir[0], dir[2]),
      pitch: Math.atan2(dir[1], Math.hypot(dir[0], dir[2])),
    },
  };
}

export function flyable(t: Track): boolean {
  const n = t.centers.length;
  for (let i = 0; i < n; i += 1) {
    const j = (i + 1) % n;
    const a = t.normals[i];
    const b = t.normals[j];
    const turn = (Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180) / Math.PI;
    if (turn > MAX_GATE_TURN_DEG || Math.abs(t.centers[j][1] - t.centers[i][1]) > MAX_GATE_CLIMB) return false;
  }
  return true;
}

function once(rand: () => number): Track {
  const u = (lo: number, hi: number) => lo + (hi - lo) * rand();
  const N = 720;
  const phi = Array.from({ length: N }, (_, i) => (2 * Math.PI * i) / N);
  let x: number[] = [];
  let z: number[] = [];
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const r0 = u(200, 320);
    const sx = u(0.75, 1.3);
    const sz = u(0.75, 1.3);
    const h = [2, 3].map((k) => ({ k, a: u(0, 0.12 / (k - 1)), p: u(0, 2 * Math.PI) }));
    const r = phi.map((f) => 1 + h.reduce((s, { k, a, p }) => s + a * Math.cos(k * f + p), 0));
    x = phi.map((f, i) => r0 * sx * r[i] * Math.cos(f));
    z = phi.map((f, i) => r0 * sz * r[i] * Math.sin(f));
    // Curvature radius of the horizontal curve, by central differences.
    let maxCurv = 0;
    for (let i = 0; i < N; i += 1) {
      const p = (i - 1 + N) % N;
      const q = (i + 1) % N;
      const dx = (x[q] - x[p]) / 2;
      const dz = (z[q] - z[p]) / 2;
      const ddx = x[q] - 2 * x[i] + x[p];
      const ddz = z[q] - 2 * z[i] + z[p];
      const curv = Math.abs(dx * ddz - dz * ddx) / Math.max((dx * dx + dz * dz) ** 1.5, 1e-9);
      maxCurv = Math.max(maxCurv, curv);
    }
    if (1 / Math.max(maxCurv, 1e-9) >= MIN_TURN_RADIUS) break;
  }
  const a1 = u(8, 22);
  const a2 = u(0, 10);
  const p1 = u(0, 2 * Math.PI);
  const p2 = u(0, 2 * Math.PI);
  let pts: Vec3[] = phi.map((f, i) => [x[i], Math.min(100, Math.max(35, 58 + a1 * Math.sin(f + p1) + a2 * Math.sin(2 * f + p2))), z[i]]);
  if (rand() < 0.5) pts = pts.reverse();
  let length = 0;
  for (let i = 0; i < pts.length; i += 1) length += norm(sub(pts[(i + 1) % pts.length], pts[i]));
  const nGates = Math.min(12, Math.max(7, Math.round(length / u(195, 225))));
  return gatesOnLine(pts, nGates, rand);
}

/** A random circuit that passes flyable(), reproducible from `seed`. */
export function randomTrack(seed: number): Track {
  const rand = mulberry32(seed);
  let t = once(rand);
  for (let i = 0; i < 100 && !flyable(t); i += 1) t = once(rand);
  return t;
}
