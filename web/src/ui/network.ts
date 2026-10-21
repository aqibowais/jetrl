const OBS_LABELS = [
  "gate x", "gate y", "gate z", "next x", "next y", "next z",
  "vel x", "vel y", "vel z", "down x", "down y", "down z",
  "pitch rt", "roll rt", "yaw rt", "throttle", "altitude", "speed",
  "aoa", "gate #", "gate dist", "g load",
];
const ACTIONS = ["THROTTLE", "PITCH", "ROLL", "YAW"];

// Diverging: cool pole, neutral midpoint, warm pole.
const NEG = [92, 200, 255];
const MID = [30, 36, 44];
const POS = [255, 176, 0];

function diverge(v: number): string {
  const t = Math.max(-1, Math.min(1, v));
  const pole = t < 0 ? NEG : POS;
  const a = Math.abs(t) ** 0.8;
  const c = MID.map((m, i) => Math.round(m + (pole[i] - m) * a));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Live view of the actor network: inputs, both hidden layers, and outputs. */
export class NetworkView {
  private readonly ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private w = 0;
  private h = 0;

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

  draw(obs: ArrayLike<number>, hidden: readonly Float32Array[], action: readonly number[]): void {
    const { ctx, w, h } = this;
    if (!w || !h) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "500 9.5px 'IBM Plex Mono', monospace";
    ctx.textBaseline = "middle";

    const labelW = 62;
    const top = 16;
    const rows = obs.length;
    const rowH = Math.max(7, Math.min(12, (h - top - 110) / rows));
    const cellW = 12;

    // Inputs.
    ctx.fillStyle = "#6b7682";
    ctx.textAlign = "left";
    ctx.fillText("IN", 0, 6);
    for (let i = 0; i < rows; i += 1) {
      const y = top + i * rowH;
      ctx.fillStyle = "#6b7682";
      ctx.fillText(OBS_LABELS[i] ?? `o${i}`, 0, y + rowH / 2);
      ctx.fillStyle = diverge(obs[i]);
      ctx.fillRect(labelW, y + 1, cellW, rowH - 2);
    }

    // Hidden layers stacked as 16 x 8 grids to the right of the inputs.
    const gridX0 = labelW + cellW + 18;
    const gridW = w - gridX0;
    const inputsBottom = top + rows * rowH;
    const cols = 16;
    const cw = gridW / cols;
    const layerH = (inputsBottom - top - 18 * Math.max(0, hidden.length - 1)) / Math.max(1, hidden.length);
    hidden.forEach((layer, li) => {
      const rowsN = Math.ceil(layer.length / cols);
      const ch = Math.min(cw, layerH / rowsN);
      const y0 = top + li * (layerH + 18);
      ctx.fillStyle = "#6b7682";
      ctx.textAlign = "left";
      ctx.fillText(`h${li + 1}`, gridX0, y0 - 9 + (li === 0 ? -1 : 0));
      for (let i = 0; i < layer.length; i += 1) {
        const cx = gridX0 + (i % cols) * cw;
        const cy = y0 + Math.floor(i / cols) * ch;
        ctx.fillStyle = diverge(layer[i]);
        ctx.fillRect(cx + 0.5, cy + 0.5, cw - 1, ch - 1);
      }
    });
    if (!hidden.length) {
      ctx.fillStyle = "#6b7682";
      ctx.textAlign = "left";
      ctx.fillText("You are flying. No network.", gridX0, top + 8);
    }

    // Outputs as centered bars, two by two under everything else.
    const outTop = inputsBottom + 22;
    ctx.fillStyle = "#6b7682";
    ctx.textAlign = "left";
    ctx.fillText("OUT", 0, outTop - 10);
    const outW = (w - 16) / 2;
    const barH = 30;
    ACTIONS.forEach((name, i) => {
      const ox = (i % 2) * (outW + 16);
      const y = outTop + 6 + Math.floor(i / 2) * (barH + 10);
      const v = action[i] ?? 0;
      ctx.fillStyle = "#8a96a3";
      ctx.textAlign = "left";
      ctx.fillText(name, ox, y);
      ctx.textAlign = "right";
      ctx.fillStyle = "#d7dde4";
      ctx.fillText((v >= 0 ? "+" : "") + v.toFixed(2), ox + outW, y);
      const by = y + 9;
      const mid = ox + outW / 2;
      ctx.fillStyle = "#161b21";
      ctx.fillRect(ox, by, outW, 8);
      ctx.fillStyle = diverge(v);
      const len = (outW / 2) * Math.max(-1, Math.min(1, v));
      ctx.fillRect(Math.min(mid, mid + len), by, Math.abs(len), 8);
      ctx.fillStyle = "#3a4552";
      ctx.fillRect(mid - 0.5, by - 2, 1, 12);
    });
  }
}
