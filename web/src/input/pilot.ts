import type { JetState } from "../sim/jet";

const STEER = ["KeyW", "KeyS", "KeyA", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"];

/**
 * Race-mode stick: the same four commands the policy outputs. With assist on,
 * left/right holds a bank angle instead of a roll rate, releasing levels
 * the wings and the flight path, and the throttle holds cruise speed.
 */
export class Pilot {
  readonly down = new Set<string>();
  assist = true;
  private stick = [0, 0, 0, 0];
  active = false;

  constructor() {
    window.addEventListener("keydown", (e) => {
      if (!this.active) return;
      if (STEER.includes(e.code)) e.preventDefault();
      this.down.add(e.code);
    });
    window.addEventListener("keyup", (e) => this.down.delete(e.code));
    window.addEventListener("blur", () => this.down.clear());
  }

  private axis(pos: string, neg: string): number {
    return (this.down.has(pos) ? 1 : 0) - (this.down.has(neg) ? 1 : 0);
  }

  command(s: JetState, dt: number): number[] {
    let throttle = this.axis("KeyW", "KeyS");
    // Flight-sim convention: stick back (down arrow) raises the nose.
    let pitch = this.axis("ArrowDown", "ArrowUp");
    let roll = this.axis("ArrowRight", "ArrowLeft");
    let yaw = this.axis("KeyD", "KeyA");

    const pad = navigator.getGamepads?.().find((g) => g && g.connected);
    if (pad) {
      const dz = (v: number) => (Math.abs(v) < 0.12 ? 0 : v);
      roll = roll || dz(pad.axes[0] ?? 0);
      pitch = pitch || dz(pad.axes[1] ?? 0);
      yaw = yaw || dz(pad.axes[2] ?? 0);
      const rt = pad.buttons[7]?.value ?? 0;
      const lt = pad.buttons[6]?.value ?? 0;
      throttle = throttle || rt - lt;
    }

    const ramp = 1 - Math.exp(-dt / 0.09);
    const target = [throttle, pitch, roll, yaw];
    for (let i = 0; i < 4; i += 1) this.stick[i] += (target[i] - this.stick[i]) * ramp;
    const out = [...this.stick];
    if (this.assist) {
      const clamp = (v: number) => Math.max(-1, Math.min(1, v));
      out[2] = clamp((this.stick[2] * 1.1 - s.bank) * 2.5 - s.rollRate * 0.2);
      if (Math.abs(pitch) < 0.05) out[1] = clamp(-s.gamma * 1.2 - s.pitchRate * 0.4);
      if (Math.abs(throttle) < 0.05) out[0] = clamp((42 - s.speed) * 0.3);
    }
    return out;
  }

  reset(): void {
    this.stick = [0, 0, 0, 0];
  }
}
