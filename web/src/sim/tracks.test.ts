import { describe, expect, it } from "vitest";
import { JetSim } from "./jet";
import { flyable, randomTrack } from "./tracks";

describe("random tracks", () => {
  it("are reproducible from the seed", () => {
    expect(randomTrack(42).centers).toEqual(randomTrack(42).centers);
    expect(randomTrack(42).centers).not.toEqual(randomTrack(43).centers);
  });

  it("match the training distribution", () => {
    for (let seed = 1; seed <= 40; seed += 1) {
      const t = randomTrack(seed);
      expect(flyable(t)).toBe(true);
      expect(t.centers.length).toBeGreaterThanOrEqual(7);
      expect(t.centers.length).toBeLessThanOrEqual(12);
      expect(t.lapLength).toBeGreaterThan(1200);
      expect(t.lapLength).toBeLessThan(2800);
      for (const c of t.centers) {
        expect(c[1]).toBeGreaterThanOrEqual(35);
        expect(c[1]).toBeLessThanOrEqual(100);
      }
      for (const n of t.normals) expect(Math.hypot(...n)).toBeCloseTo(1, 6);
    }
  });

  it("load into the simulator and start on the approach to gate 1", () => {
    const sim = new JetSim(randomTrack(7));
    sim.reset();
    expect(sim.state.nextGate).toBe(0);
    const d = sim.gateDistance();
    expect(d).toBeGreaterThan(30);
    expect(d).toBeLessThan(120);
  });
});
