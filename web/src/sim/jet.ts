/**
 * TypeScript port of jet_env.py. Same constants, same step order, float64.
 * `parity.test.ts` replays a Python-recorded episode through this file and
 * checks the state step by step, so keep the two in lockstep when editing.
 */

export const SIM_HZ = 30;
export const DT = 1 / SIM_HZ;
export const MAX_STEPS = 3000;

const GRAVITY = 9.81;
const CRUISE_SPEED = 42.0;
const THRUST_MAX = 18.0;
const DRAG_K = 0.0042;
const INDUCED_K = 1.2;
const AOA_1G = 0.06;
const AOA_MAX = 0.32;
const N_MAX = 6.0;
const N_MIN = -1.5;
const PULL_G = 3.5;
const PUSH_G = 1.5;
const MAX_ROLL_RATE = 2.2;
const MAX_RUDDER_RATE = 0.25;
const MAX_BANK = 1.35;
const MAX_GAMMA = 1.2;
const RATE_TAU = 0.12;
const MIN_SPEED = 8.0;
const MAX_SPEED = 85.0;
const GROUND_Y = 1.5;
const CEILING_Y = 280.0;
const PHI_SCALE = 0.06;
const GATE_REWARD = 12.0;
const LAP_BONUS = 50.0;
const PAR_TIME = 52.0;
const CRASH_PENALTY = 40.0;
const STEP_PENALTY = 0.002;
const MAX_TRACK_DISTANCE = 420.0;
const GATE_TIMEOUT = 20.0;
const STALL_PENALTY = 20.0;

export const OBS_DIM = 22;

export type Vec3 = [number, number, number];

export interface Track {
  centers: Vec3[];
  normals: Vec3[];
  radius: number;
  centerline: Vec3[];
  lapLength: number;
  start: { position: Vec3; yaw: number; pitch: number };
}

export interface JetState {
  position: Vec3;
  velocity: Vec3;
  /** Flight path: heading, climb angle, bank. pitch/yaw/roll are the body attitude. */
  heading: number;
  gamma: number;
  bank: number;
  pull: number;
  rudder: number;
  pitch: number;
  roll: number;
  yaw: number;
  pitchRate: number;
  rollRate: number;
  yawRate: number;
  throttle: number;
  speed: number;
  aoa: number;
  gForce: number;
  nextGate: number;
  cleared: boolean[];
  prevDist: number;
  steps: number;
  time: number;
  lapStart: number | null;
  lapTime: number | null;
  lapRunning: boolean;
  episodeReward: number;
  trackDist: number;
  lastProgress: number;
}

export type EndReason = "" | "crash_ground" | "crash_bounds" | "lap_complete" | "stalled" | "timeout";

export interface StepResult {
  reward: number;
  terminated: boolean;
  truncated: boolean;
  reason: EndReason;
  passedGate: number | null;
}

const TAU = Math.PI * 2;

export function wrapAngle(angle: number): number {
  // Python's % is floored; JS % truncates. Match numpy.
  let r = (angle + Math.PI) % TAU;
  if (r !== 0 && r < 0) r += TAU;
  return r - Math.PI;
}

const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a: Vec3) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** Aircraft axes in world coordinates. +roll drops the right wing. */
export function bodyAxes(yaw: number, pitch: number, roll: number): { forward: Vec3; right: Vec3; up: Vec3 } {
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const forward: Vec3 = [sy * cp, sp, cy * cp];
  const right0: Vec3 = [cy, 0, -sy];
  const up0 = cross(forward, right0);
  const right: Vec3 = [right0[0] * cr - up0[0] * sr, right0[1] * cr - up0[1] * sr, right0[2] * cr - up0[2] * sr];
  const up: Vec3 = [up0[0] * cr + right0[0] * sr, up0[1] * cr + right0[1] * sr, up0[2] * cr + right0[2] * sr];
  return { forward, right, up };
}

/** Yaw, pitch and roll (as used by bodyAxes) of a body frame. */
export function attitude(forward: Vec3, right: Vec3): [number, number, number] {
  const yaw = Math.atan2(forward[0], forward[2]);
  const pitch = Math.asin(clip(forward[1], -1, 1));
  const right0: Vec3 = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const up0 = cross(forward, right0);
  const roll = Math.atan2(-dot(right, up0), dot(right, right0));
  return [yaw, pitch, roll];
}

function lag(current: number, target: number, dt: number, tau: number): number {
  const alpha = 1 - Math.exp(-dt / tau);
  return current + (target - current) * alpha;
}

export function distanceToCenterline(p: Vec3, line: readonly Vec3[]): number {
  let best = Infinity;
  const n = line.length;
  for (let i = 0; i < n; i += 1) {
    const a = line[i];
    const b = line[(i + 1) % n];
    const ab = sub(b, a);
    const denom = Math.max(dot(ab, ab), 1e-8);
    const t = clip(dot(sub(p, a), ab) / denom, 0, 1);
    const d = norm([a[0] + t * ab[0] - p[0], a[1] + t * ab[1] - p[1], a[2] + t * ab[2] - p[2]]);
    if (d < best) best = d;
  }
  return best;
}

export class JetSim {
  readonly track: Track;
  readonly nGates: number;
  state: JetState;

  constructor(track: Track) {
    this.track = track;
    this.nGates = track.centers.length;
    this.state = this.blank();
  }

  private blank(): JetState {
    return {
      position: [0, 0, 0],
      velocity: [0, 0, 0],
      heading: 0,
      gamma: 0,
      bank: 0,
      pull: 0,
      rudder: 0,
      pitch: 0,
      roll: 0,
      yaw: 0,
      pitchRate: 0,
      rollRate: 0,
      yawRate: 0,
      throttle: 0.48,
      speed: 0,
      aoa: 0,
      gForce: 1,
      nextGate: 0,
      cleared: new Array(this.nGates).fill(false),
      prevDist: 0,
      steps: 0,
      time: 0,
      lapStart: null,
      lapTime: null,
      lapRunning: false,
      episodeReward: 0,
      trackDist: 0,
      lastProgress: 0,
    };
  }

  /** Python's reset draws small random offsets. `jitter` reproduces that with any RNG. */
  reset(jitter: () => number = () => 0.5): void {
    const s = this.blank();
    const u = (lo: number, hi: number) => lo + (hi - lo) * jitter();
    const start = this.track.start;
    s.position = [start.position[0] + u(-6, 6), start.position[1] + u(-6, 6), start.position[2] + u(-6, 6)];
    s.position[1] = clip(s.position[1], 28, 120);
    s.heading = wrapAngle(start.yaw + u(-0.06, 0.06));
    s.gamma = start.pitch + u(-0.04, 0.04);
    s.bank = u(-0.04, 0.04);
    s.speed = CRUISE_SPEED + u(-2, 2);
    s.throttle = clip(0.48 + u(-0.05, 0.05), 0, 1);
    this.updateFrame(s, 1);
    this.state = s;
    s.prevDist = this.gateDistance();
    s.trackDist = distanceToCenterline(s.position, this.track.centerline);
  }

  private updateFrame(s: JetState, q: number): void {
    const { forward: path, right, up } = bodyAxes(s.heading, s.gamma, s.bank);
    s.velocity = [path[0] * s.speed, path[1] * s.speed, path[2] * s.speed];
    s.aoa = clip((s.gForce * AOA_1G) / Math.max(q, 1e-3), -0.15, AOA_MAX);
    const ca = Math.cos(s.aoa);
    const sa = Math.sin(s.aoa);
    const forward: Vec3 = [path[0] * ca + up[0] * sa, path[1] * ca + up[1] * sa, path[2] * ca + up[2] * sa];
    [s.yaw, s.pitch, s.roll] = attitude(forward, right);
  }

  /** Load a full state, used by the parity test and by checkpoint scrubbing. */
  load(state: JetState): void {
    this.state = { ...state, position: [...state.position], velocity: [...state.velocity], cleared: [...state.cleared] };
  }

  step(raw: ArrayLike<number>): StepResult {
    const s = this.state;
    const action = [0, 1, 2, 3].map((i) => clip(Number(raw[i] ?? 0), -1, 1));
    s.steps += 1;
    s.time += DT;

    s.throttle = clip(s.throttle + action[0] * 0.55 * DT, 0, 1);
    s.pull = lag(s.pull, action[1], DT, RATE_TAU);
    s.rollRate = lag(s.rollRate, action[2] * MAX_ROLL_RATE, DT, RATE_TAU);
    s.rudder = lag(s.rudder, action[3] * MAX_RUDDER_RATE, DT, RATE_TAU);

    const oldBank = s.bank;
    s.bank = clip(s.bank + s.rollRate * DT, -MAX_BANK, MAX_BANK);
    s.rollRate = (s.bank - oldBank) / DT;

    const v = Math.max(s.speed, MIN_SPEED);
    const q = (v / CRUISE_SPEED) ** 2;
    const nAvail = Math.min((AOA_MAX / AOA_1G) * q, N_MAX);
    const trim = Math.cos(s.gamma) / Math.cos(s.bank);
    const n = clip(trim + s.pull * (s.pull > 0 ? PULL_G : PUSH_G), N_MIN, nAvail);

    const gammaDot = (GRAVITY / v) * (n * Math.cos(s.bank) - Math.cos(s.gamma));
    const headingDot = ((GRAVITY / v) * n * Math.sin(s.bank)) / Math.cos(s.gamma) + s.rudder;
    const oldGamma = s.gamma;
    s.gamma = clip(s.gamma + gammaDot * DT, -MAX_GAMMA, MAX_GAMMA);
    s.pitchRate = (s.gamma - oldGamma) / DT;
    s.yawRate = headingDot;
    s.heading = wrapAngle(s.heading + headingDot * DT);

    const drag = DRAG_K * v * v + (INDUCED_K * n * n) / q;
    const accel = s.throttle * THRUST_MAX - drag - GRAVITY * Math.sin(s.gamma);
    s.speed = clip(s.speed + accel * DT, MIN_SPEED, MAX_SPEED);
    s.gForce = n;
    this.updateFrame(s, q);

    const previous: Vec3 = [...s.position];
    s.position = [s.position[0] + s.velocity[0] * DT, s.position[1] + s.velocity[1] * DT, s.position[2] + s.velocity[2] * DT];
    s.trackDist = distanceToCenterline(s.position, this.track.centerline);

    let reward = -STEP_PENALTY;
    let terminated = false;
    let truncated = false;
    let reason: EndReason = "";
    let passedGate: number | null = null;

    const belowGround = s.position[1] < GROUND_Y;
    const outOfBounds = s.trackDist > MAX_TRACK_DISTANCE || s.position[1] > CEILING_Y;
    const invalid = !s.position.every(Number.isFinite) || !s.velocity.every(Number.isFinite);

    if (belowGround || outOfBounds || invalid) {
      if (invalid) {
        s.position = s.position.map((v) => (Number.isFinite(v) ? v : 0)) as Vec3;
        s.velocity = [0, 0, 0];
      }
      reward -= CRASH_PENALTY;
      terminated = true;
      reason = belowGround ? "crash_ground" : "crash_bounds";
    } else {
      const passed = this.checkGate(previous, s.position);
      if (passed === null) {
        const dist = this.gateDistance();
        reward += PHI_SCALE * clip(s.prevDist - dist, -8, 8);
        s.prevDist = dist;
      } else {
        passedGate = passed.index;
        const centering = 1 - passed.offset / this.track.radius;
        reward += GATE_REWARD * (0.35 + 0.65 * centering);
        this.advanceGate(passed.index);
        s.prevDist = this.gateDistance();
      }
      if (s.cleared.every(Boolean)) {
        const elapsed = Math.max(s.lapTime ?? s.time, 1);
        reward += LAP_BONUS * clip(PAR_TIME / elapsed, 0.45, 2.4);
        terminated = true;
        reason = "lap_complete";
      } else if (s.time - s.lastProgress > GATE_TIMEOUT) {
        reward -= STALL_PENALTY;
        terminated = true;
        reason = "stalled";
      } else if (s.steps >= MAX_STEPS) {
        truncated = true;
        reason = "timeout";
      }
    }
    s.episodeReward += reward;
    return { reward, terminated, truncated, reason, passedGate };
  }

  gateDistance(): number {
    const s = this.state;
    return norm(sub(this.track.centers[s.nextGate], s.position));
  }

  private checkGate(previous: Vec3, current: Vec3): { index: number; offset: number } | null {
    const index = this.state.nextGate;
    const center = this.track.centers[index];
    const normal = this.track.normals[index];
    const d0 = dot(sub(previous, center), normal);
    const d1 = dot(sub(current, center), normal);
    if (!(d0 < 0 && d1 >= 0)) return null;
    const denom = d0 - d1;
    if (Math.abs(denom) < 1e-8) return null;
    const t = d0 / denom;
    if (t < 0 || t > 1) return null;
    const point: Vec3 = [
      previous[0] + t * (current[0] - previous[0]),
      previous[1] + t * (current[1] - previous[1]),
      previous[2] + t * (current[2] - previous[2]),
    ];
    const off = sub(point, center);
    const along = dot(off, normal);
    const offset = norm([off[0] - normal[0] * along, off[1] - normal[1] * along, off[2] - normal[2] * along]);
    if (offset > this.track.radius) return null;
    return { index, offset };
  }

  private advanceGate(index: number): void {
    const s = this.state;
    s.cleared[index] = true;
    s.lastProgress = s.time;
    if (index === 0 && !s.lapRunning) {
      s.lapRunning = true;
      s.lapStart = s.time;
      s.lapTime = 0;
    }
    if (s.lapRunning && s.lapStart !== null) s.lapTime = s.time - s.lapStart;
    s.nextGate = (index + 1) % this.nGates;
  }

  /** Lap clock that keeps running between gates, for the HUD. */
  liveLapTime(): number | null {
    const s = this.state;
    if (!s.lapRunning || s.lapStart === null) return null;
    if (s.cleared.every(Boolean)) return s.lapTime;
    return s.time - s.lapStart;
  }

  /** maskGateIndex: generalist policies were trained with the gate-index feature zeroed. */
  observation(out: Float64Array = new Float64Array(OBS_DIM), maskGateIndex = false): Float64Array {
    const s = this.state;
    const { forward, right, up } = bodyAxes(s.yaw, s.pitch, s.roll);
    const n = this.nGates;
    const rel = sub(this.track.centers[s.nextGate], s.position);
    const follow = sub(this.track.centers[(s.nextGate + 1) % n], s.position);
    const down: Vec3 = [0, -1, 0];
    const dist = norm(rel);
    const values = [
      dot(rel, right) / 200,
      dot(rel, up) / 200,
      dot(rel, forward) / 200,
      dot(follow, right) / 250,
      dot(follow, up) / 250,
      dot(follow, forward) / 250,
      dot(s.velocity, right) / 60,
      dot(s.velocity, up) / 60,
      dot(s.velocity, forward) / 60,
      dot(down, right),
      dot(down, up),
      dot(down, forward),
      s.pitchRate / 2,
      s.rollRate / 2,
      s.yawRate / 2,
      s.throttle,
      s.position[1] / 120,
      s.speed / 70,
      clip(s.aoa, -1, 1),
      maskGateIndex ? 0 : s.nextGate / n,
      dist / 300,
      s.gForce / 4,
    ];
    for (let i = 0; i < OBS_DIM; i += 1) out[i] = clip(Math.fround(values[i]), -10, 10);
    return out;
  }
}
