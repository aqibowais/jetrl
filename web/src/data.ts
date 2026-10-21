import type { Track, Vec3 } from "./sim/jet";
import { Policy, type PolicyJson } from "./sim/policy";

export interface EvalStats {
  episodes: number;
  lap_rate: number;
  best_lap: number | null;
  mean_lap: number | null;
  mean_gates: number;
  mean_reward: number;
  mean_airtime: number;
  outcomes: Record<string, number>;
}

export interface CheckpointMeta {
  id: string;
  episode: number;
  timesteps: number;
  train_mean_reward: number | null;
  eval: EvalStats;
}

/** Results on procedurally generated tracks no policy trained on. */
export interface UnseenStats {
  tracks: number;
  lap_rate: number;
  mean_lap: number | null;
  /** Share of gates cleared, averaged over tracks. */
  gate_share: number;
  outcomes: Record<string, number>;
}

export interface AblationRun {
  seed: number;
  timesteps: number;
  lap_rate: number;
  mean_lap: number | null;
  gate_share: number;
}

export interface Ablation {
  id: string;
  label: string;
  detail: string;
  runs: AblationRun[];
}

export interface Manifest {
  best: string;
  eval_seeds: number;
  checkpoints: CheckpointMeta[];
  generalist?: { id: string; episode: number; timesteps: number; fixed: EvalStats; unseen: UnseenStats };
  specialist_unseen?: UnseenStats;
  /** A browser track seed where the specialist fails and the generalist laps, for the story. */
  story_seed?: number;
  ablation?: Ablation[];
  /** test_env.py's hand-written pilot on the same seeds. */
  heuristic?: EvalStats;
}

export interface CurveBin {
  episode: number;
  timesteps: number;
  reward: number;
  reward_lo: number;
  reward_hi: number;
  gates: number;
  laps: number;
}

export interface Curve {
  episodes: number;
  bins: CurveBin[];
}

interface TrackJson {
  gates: { center: Vec3; normal: Vec3; radius: number; yaw: number; pitch: number }[];
  centerline: Vec3[];
  lap_length: number;
  start: { position: Vec3; yaw: number; pitch: number };
}

export interface TrackData extends Track {
  yaws: number[];
  pitches: number[];
}

async function json<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return (await res.json()) as T;
}

export async function loadTrack(): Promise<TrackData> {
  const raw = await json<TrackJson>("data/track.json");
  return {
    centers: raw.gates.map((g) => g.center),
    normals: raw.gates.map((g) => g.normal),
    yaws: raw.gates.map((g) => g.yaw),
    pitches: raw.gates.map((g) => g.pitch),
    radius: raw.gates[0].radius,
    centerline: raw.centerline,
    lapLength: raw.lap_length,
    start: raw.start,
  };
}

export const loadManifest = () => json<Manifest>("data/manifest.json");
export const loadCurve = () => json<Curve>("data/curve.json").catch(() => ({ episodes: 0, bins: [] }) as Curve);

const cache = new Map<string, Promise<Policy>>();
export function loadPolicy(id: string): Promise<Policy> {
  let hit = cache.get(id);
  if (!hit) {
    hit = json<PolicyJson>(`data/policies/${id}.json`).then((p) => new Policy(p));
    cache.set(id, hit);
  }
  return hit;
}
