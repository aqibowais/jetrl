import { bodyAxes, type JetState } from "../sim/jet";

const DEG = 180 / Math.PI;
const FONT = "500 12px 'IBM Plex Mono', monospace";
const FONT_BIG = "600 16px 'IBM Plex Mono', monospace";

export interface HudInfo {
  gate: number;
  gates: number;
  gateDistance: number;
}

/** Head-up display: boresight, flight-path marker, pitch ladder, heading tape, speed and altitude. */
export class Instruments {
  private readonly ctx: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private dpr = 1;
  /** Pixels on the right covered by a panel; the HUD centers in what is left. */
  inset = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context");
    this.ctx = ctx;
  }

  resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
  }

  clear(): void {
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(s: JetState, color: string, compact: boolean, info: HudInfo | null = null): void {
    const { ctx } = this;
    this.clear();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    // A dark halo keeps green symbology readable against bright sky and snow.
    ctx.shadowColor = "rgba(0, 0, 0, 0.55)";
    ctx.shadowBlur = 3;
    const vw = Math.max(320, this.w - this.inset);
    const cx = vw / 2;
    const cy = this.h * 0.45;
    const pxPerDeg = Math.max(6, Math.min(vw, this.h) / 70);

    // Flight-path marker: velocity direction relative to the nose.
    const { forward, right, up } = bodyAxes(s.yaw, s.pitch, s.roll);
    const v = s.velocity;
    const vf = v[0] * forward[0] + v[1] * forward[1] + v[2] * forward[2];
    const vr = v[0] * right[0] + v[1] * right[1] + v[2] * right[2];
    const vu = v[0] * up[0] + v[1] * up[1] + v[2] * up[2];
    const beta = Math.atan2(vr, Math.max(vf, 1)) * DEG;
    const alpha = Math.atan2(vu, Math.max(vf, 1)) * DEG;
    const fx = cx + Math.max(-12, Math.min(12, beta)) * pxPerDeg;
    const fy = cy - Math.max(-12, Math.min(12, alpha)) * pxPerDeg;

    // Pitch ladder, rolled around the flight-path marker.
    ctx.save();
    ctx.beginPath();
    const clipR = Math.min(vw, this.h) * 0.3;
    ctx.rect(cx - clipR, cy - clipR, clipR * 2, clipR * 2);
    ctx.clip();
    ctx.translate(fx, fy);
    ctx.rotate(-s.roll);
    const pitchDeg = s.pitch * DEG;
    ctx.font = FONT;
    ctx.textBaseline = "middle";
    for (let d = -60; d <= 60; d += 10) {
      const y = (pitchDeg - d) * pxPerDeg;
      if (Math.abs(y) > clipR) continue;
      const half = d === 0 ? clipR * 0.95 : 70;
      const gap = d === 0 ? 40 : 28;
      ctx.globalAlpha = d === 0 ? 0.9 : 0.7;
      ctx.beginPath();
      if (d < 0) ctx.setLineDash([7, 5]);
      ctx.moveTo(-half, y);
      ctx.lineTo(-gap, y);
      ctx.moveTo(gap, y);
      ctx.lineTo(half, y);
      if (d !== 0) {
        const tick = d > 0 ? 8 : -8;
        ctx.moveTo(-half, y);
        ctx.lineTo(-half, y + tick);
        ctx.moveTo(half, y);
        ctx.lineTo(half, y + tick);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (d !== 0) {
        ctx.textAlign = "right";
        ctx.fillText(String(Math.abs(d)), -half - 6, y);
        ctx.textAlign = "left";
        ctx.fillText(String(Math.abs(d)), half + 6, y);
      }
    }
    ctx.restore();
    ctx.globalAlpha = 1;

    // Boresight (the nose) and the flight-path marker.
    ctx.beginPath();
    ctx.moveTo(cx - 16, cy);
    ctx.lineTo(cx - 8, cy);
    ctx.lineTo(cx - 4, cy + 6);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + 4, cy + 6);
    ctx.lineTo(cx + 8, cy);
    ctx.lineTo(cx + 16, cy);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(fx, fy, 7, 0, Math.PI * 2);
    ctx.moveTo(fx - 7, fy);
    ctx.lineTo(fx - 20, fy);
    ctx.moveTo(fx + 7, fy);
    ctx.lineTo(fx + 20, fy);
    ctx.moveTo(fx, fy - 7);
    ctx.lineTo(fx, fy - 14);
    ctx.stroke();

    if (compact) {
      ctx.shadowBlur = 0;
      return;
    }

    // Heading tape.
    this.heading(cx, 70, s.yaw * DEG);
    // Speed (knots) and altitude (meters) boxes.
    const side = Math.min(300, vw * 0.3);
    this.box(cx - side, cy, Math.round(s.speed * 1.944), "KTS", "right");
    this.box(cx + side, cy, Math.round(s.position[1]), "ALT M", "left");

    // Data blocks.
    ctx.font = FONT;
    ctx.textAlign = "left";
    const lx = cx - side - 30;
    const ly = cy + 70;
    const lines = [`G ${s.gForce.toFixed(1)}`, `α ${(s.aoa * DEG).toFixed(1)}°`];
    lines.forEach((t, i) => ctx.fillText(t, lx, ly + i * 17));
    const rx = cx + side - 10;
    const right2 = info ? [`GATE ${info.gate + 1}/${info.gates}`, `${Math.round(info.gateDistance)} M`] : [];
    right2.forEach((t, i) => ctx.fillText(t, rx, ly + i * 17));

    // Throttle gauge, bottom right.
    const gx = vw - 40;
    const gy = this.h - 190;
    ctx.globalAlpha = 0.85;
    ctx.strokeRect(gx, gy, 14, 140);
    ctx.globalAlpha = 0.35;
    ctx.fillRect(gx, gy, 14, 140);
    ctx.globalAlpha = 0.95;
    ctx.fillRect(gx, gy + 140 * (1 - s.throttle), 14, 140 * s.throttle);
    ctx.globalAlpha = 1;
    ctx.textAlign = "center";
    ctx.fillText("THR", gx + 7, gy - 10);
    ctx.shadowBlur = 0;
  }

  private box(x: number, y: number, value: number, label: string, side: "left" | "right"): void {
    const { ctx } = this;
    const bw = 64;
    const bx = side === "right" ? x - bw : x;
    ctx.strokeRect(bx, y - 13, bw, 26);
    ctx.font = FONT_BIG;
    ctx.textAlign = "center";
    ctx.fillText(String(value), bx + bw / 2, y + 1);
    ctx.font = FONT;
    ctx.globalAlpha = 0.85;
    ctx.fillText(label, bx + bw / 2, y - 24);
    ctx.globalAlpha = 1;
  }

  private heading(cx: number, y: number, yawDeg: number): void {
    const { ctx } = this;
    const hdg = ((yawDeg % 360) + 360) % 360;
    const width = Math.min(340, this.w * 0.45);
    const pxPer = width / 60;
    ctx.save();
    ctx.beginPath();
    ctx.rect(cx - width / 2, y - 20, width, 34);
    ctx.clip();
    ctx.font = FONT;
    ctx.textAlign = "center";
    const first = Math.floor((hdg - 30) / 5) * 5;
    for (let d = first; d <= hdg + 30; d += 5) {
      const x = cx + (d - hdg) * pxPer;
      const major = d % 10 === 0;
      ctx.beginPath();
      ctx.moveTo(x, y + 10);
      ctx.lineTo(x, y + (major ? 0 : 5));
      ctx.stroke();
      if (major) ctx.fillText(String(((d % 360) + 360) % 360 / 10).padStart(2, "0"), x, y - 8);
    }
    ctx.restore();
    ctx.beginPath();
    ctx.moveTo(cx, y + 12);
    ctx.lineTo(cx - 5, y + 20);
    ctx.lineTo(cx + 5, y + 20);
    ctx.closePath();
    ctx.stroke();
  }
}
