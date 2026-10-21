import type { Track, Vec3 } from "../sim/jet";

export interface MapJet {
  position: Vec3;
  color: string;
  focus: boolean;
}

/**
 * Top-down track map. Oriented like the 3D overview camera: sim +x is screen
 * right, sim +z is screen up.
 */
export class Minimap {
  private readonly ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private w = 0;
  private h = 0;
  private bounds = { minX: 0, maxX: 1, minZ: 0, maxZ: 1 };

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
  }

  setTrack(t: Track): void {
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const p of t.centerline) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minZ = Math.min(minZ, p[2]);
      maxZ = Math.max(maxZ, p[2]);
    }
    this.bounds = { minX, maxX, minZ, maxZ };
  }

  draw(t: Track, next: number, cleared: readonly boolean[], jets: MapJet[]): void {
    const { ctx, w, h } = this;
    if (!w) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pad = 14;
    const b = this.bounds;
    const span = Math.max(b.maxX - b.minX, b.maxZ - b.minZ, 1);
    const s = (Math.min(w, h) - pad * 2) / span;
    const ox = w / 2 - ((b.minX + b.maxX) / 2) * s;
    const oz = h / 2 + ((b.minZ + b.maxZ) / 2) * s;
    const X = (p: Vec3) => ox + p[0] * s;
    const Y = (p: Vec3) => oz - p[2] * s;

    ctx.strokeStyle = "#2a3440";
    ctx.lineWidth = 2;
    ctx.beginPath();
    t.centerline.forEach((p, i) => (i ? ctx.lineTo(X(p), Y(p)) : ctx.moveTo(X(p), Y(p))));
    ctx.closePath();
    ctx.stroke();

    ctx.font = "500 9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    t.centers.forEach((c, i) => {
      const n = t.normals[i];
      // Gate drawn as a short bar across the track.
      const px = -n[2];
      const pz = n[0];
      const half = t.radius * s * 1.2 + 2;
      ctx.strokeStyle = i === next ? "#ffb000" : cleared[i] ? "#4cc38a" : "#5a6774";
      ctx.lineWidth = i === next ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(X(c) - px * half, Y(c) + pz * half);
      ctx.lineTo(X(c) + px * half, Y(c) - pz * half);
      ctx.stroke();
      ctx.fillStyle = i === next ? "#ffb000" : "#7d8792";
      ctx.fillText(String(i + 1), X(c) + n[0] * 10, Y(c) - n[2] * 10 - 8);
    });

    for (const j of [...jets].sort((a, b2) => Number(a.focus) - Number(b2.focus))) {
      ctx.fillStyle = j.color;
      ctx.beginPath();
      ctx.arc(X(j.position), Y(j.position), j.focus ? 4.5 : 3, 0, Math.PI * 2);
      ctx.fill();
      if (j.focus) {
        ctx.strokeStyle = "#07090c";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }
}
