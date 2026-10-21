import type { EndReason, JetState, StepResult } from "../sim/jet";

const DEG = 180 / Math.PI;

export interface Reading {
  gate: number;
  distance: number;
  /** Degrees right (+) or left (-) of the nose. */
  bearing: number;
  /** Meters above (+) or below (-) the jet. */
  height: number;
}

/** Where the next gate is, from the policy's own observation (body frame). */
export function reading(obs: ArrayLike<number>, s: JetState): Reading {
  const right = obs[0] * 200;
  const up = obs[1] * 200;
  const fwd = obs[2] * 200;
  return { gate: s.nextGate, distance: Math.hypot(right, up, fwd), bearing: Math.atan2(right, fwd) * DEG, height: up };
}

const word = (v: number, pos: string, neg: string, hold: string, dz = 0.25) => (v > dz ? pos : v < -dz ? neg : hold);

/** The four network outputs in words. */
export function decisions(a: readonly number[]): { label: string; value: number; text: string }[] {
  return [
    { label: "Throttle", value: a[0], text: word(a[0], "more power", "less power", "hold") },
    { label: "Pitch", value: a[1], text: word(a[1], "pull", "push", "hold path") },
    { label: "Roll", value: a[2], text: word(a[2], "roll right", "roll left", "hold bank") },
    { label: "Rudder", value: a[3], text: word(a[3], "right", "left", "centered") },
  ];
}

export function narrate(r: Reading, a: readonly number[]): string {
  const parts = [r.distance < 40 ? "close" : `${Math.round(r.distance)} m`];
  if (Math.abs(r.bearing) > 4) parts.push(`${Math.abs(Math.round(r.bearing))}° ${r.bearing > 0 ? "right" : "left"}`);
  if (Math.abs(r.height) > 4) parts.push(`${Math.abs(Math.round(r.height))} m ${r.height > 0 ? "up" : "down"}`);
  const doing = decisions(a)
    .filter((d) => Math.abs(d.value) > 0.25 && d.label !== "Rudder" && d.label !== "Throttle")
    .map((d) => d.text);
  const verb = doing.length ? doing.join(" and ") : "holding";
  return `Gate ${r.gate + 1}: ${parts.join(", ")}. ${verb[0].toUpperCase()}${verb.slice(1)}.`;
}

export interface RewardEvent {
  t: number;
  value: number;
  text: string;
  tone: "good" | "bad" | "info";
}

const ENDING: Partial<Record<EndReason, string>> = {
  crash_ground: "hit the ground",
  crash_bounds: "left the airspace",
  stalled: "20 s without a gate",
  lap_complete: "lap complete",
};

/** Gate passes and endings as they happen; the small progress reward summed once a second. */
export class RewardLedger {
  events: RewardEvent[] = [];
  private progress = 0;
  private since = 0;

  reset(): void {
    this.events = [];
    this.progress = 0;
    this.since = 0;
  }

  record(t: number, r: StepResult, s: JetState, radius: number, dt: number): void {
    void radius;
    if (r.passedGate !== null) {
      this.flush(t);
      if (r.reason === "lap_complete") {
        // The last gate and the lap bonus land on the same step.
        this.push({ t, value: r.reward, text: `gate ${r.passedGate + 1} and the lap, with the pace bonus`, tone: "good" });
        return;
      }
      const off = gateOffset(r.reward);
      this.push({ t, value: r.reward, text: `gate ${r.passedGate + 1} passed${off !== null ? `, ${off}% centered` : ""}`, tone: "good" });
      return;
    }
    if (r.terminated || r.truncated) {
      this.flush(t);
      this.push({ t, value: r.reward, text: ENDING[r.reason] ?? r.reason, tone: r.reason === "lap_complete" ? "good" : "bad" });
      return;
    }
    this.progress += r.reward;
    this.since += dt;
    if (this.since >= 1) this.flush(t, s);
  }

  private flush(t: number, s?: JetState): void {
    if (Math.abs(this.progress) > 0.05) {
      const toward = this.progress > 0;
      this.push({
        t,
        value: this.progress,
        text: toward ? `closing on gate ${(s?.nextGate ?? 0) + 1}` : `drifting away from gate ${(s?.nextGate ?? 0) + 1}`,
        tone: toward ? "info" : "bad",
      });
    }
    this.progress = 0;
    this.since = 0;
  }

  private push(e: RewardEvent): void {
    this.events.unshift(e);
    if (this.events.length > 6) this.events.pop();
  }
}

/** Gate reward is 12 x (0.35 + 0.65 x centering) plus a sliver of step penalty; invert it. */
function gateOffset(reward: number): number | null {
  const centering = (reward / 12 - 0.35) / 0.65;
  if (!Number.isFinite(centering) || centering < -0.1 || centering > 1.1) return null;
  return Math.round(Math.max(0, Math.min(1, centering)) * 100);
}
