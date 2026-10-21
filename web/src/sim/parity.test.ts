import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { JetSim, OBS_DIM, wrapAngle, type JetState, type Track } from "./jet";
import { Policy, type LayerJson } from "./policy";

interface Frame {
  obs: number[];
  action: number[];
  reward: number;
  state: JetState;
}

const here = new URL(".", import.meta.url);
const fixture = JSON.parse(readFileSync(new URL("fixtures/parity.json", here), "utf-8")) as {
  initial: JetState;
  frames: Frame[];
  layers: LayerJson[];
};
const raw = JSON.parse(readFileSync(new URL("../../public/data/track.json", here), "utf-8"));
const track: Track = {
  centers: raw.gates.map((g: { center: number[] }) => g.center),
  normals: raw.gates.map((g: { normal: number[] }) => g.normal),
  radius: raw.gates[0].radius,
  centerline: raw.centerline,
  lapLength: raw.lap_length,
  start: raw.start,
};

describe("TypeScript port matches jet_env.py", () => {
  it("wraps angles like numpy's floored modulo", () => {
    expect(wrapAngle(Math.PI + 0.1)).toBeCloseTo(-Math.PI + 0.1, 12);
    expect(wrapAngle(-Math.PI - 0.1)).toBeCloseTo(Math.PI - 0.1, 12);
    expect(wrapAngle(0.3)).toBeCloseTo(0.3, 15);
  });

  it("reproduces every recorded step from the Python state", () => {
    const sim = new JetSim(track);
    let prev = fixture.initial;
    let worst = 0;
    for (const frame of fixture.frames) {
      // Restart each step from Python's state, so one-ulp libm differences
      // between numpy and V8 cannot accumulate into a chaotic divergence.
      sim.load(prev);
      const obs = sim.observation();
      for (let i = 0; i < OBS_DIM; i += 1) expect(obs[i]).toBeCloseTo(frame.obs[i], 5);
      const result = sim.step(frame.action);
      expect(result.reward).toBeCloseTo(frame.reward, 8);
      const s = sim.state;
      for (let i = 0; i < 3; i += 1) {
        worst = Math.max(worst, Math.abs(s.position[i] - frame.state.position[i]));
        expect(s.velocity[i]).toBeCloseTo(frame.state.velocity[i], 8);
      }
      expect(s.nextGate).toBe(frame.state.nextGate);
      expect(s.cleared).toEqual(frame.state.cleared);
      prev = frame.state;
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it("free-runs the recorded actions without drifting", () => {
    const sim = new JetSim(track);
    sim.load(fixture.initial);
    const n = Math.min(300, fixture.frames.length);
    for (let i = 0; i < n; i += 1) sim.step(fixture.frames[i].action);
    const want = fixture.frames[n - 1].state.position;
    const err = Math.hypot(...sim.state.position.map((v, i) => v - want[i]));
    expect(err).toBeLessThan(1e-3);
  });

  it("policy forward pass matches torch", () => {
    const policy = new Policy({ id: "fixture", layers: fixture.layers });
    let worst = 0;
    for (const frame of fixture.frames) {
      const action = policy.act(frame.obs);
      for (let i = 0; i < 4; i += 1) worst = Math.max(worst, Math.abs(action[i] - frame.action[i]));
    }
    expect(worst).toBeLessThan(1e-4);
  });
});
