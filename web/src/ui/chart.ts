const NS = "http://www.w3.org/2000/svg";

export interface LineChartSpec {
  x: number[];
  y: number[];
  lo?: number[];
  hi?: number[];
  color: string;
  xFormat: (v: number) => string;
  yFormat: (v: number) => string;
  yLabel: string;
  yMin?: number;
  yMax?: number;
  /** Vertical markers, e.g. exported checkpoints. */
  marks?: { x: number; label: string; active?: boolean }[];
  onMark?: (index: number) => void;
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** Single-series line with optional band, crosshair tooltip, and clickable marks. */
export function lineChart(host: HTMLElement, spec: LineChartSpec): void {
  host.replaceChildren();
  host.classList.add("chart");
  const width = host.clientWidth || 640;
  const height = host.clientHeight || 220;
  const m = { l: 52, r: 14, t: 14, b: 26 };
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const xs = spec.x;
  const x0 = xs[0] ?? 0;
  const x1 = xs[xs.length - 1] ?? 1;
  const all = [...spec.y, ...(spec.lo ?? []), ...(spec.hi ?? [])];
  const yMin = spec.yMin ?? Math.min(...all);
  const yMax = spec.yMax ?? Math.max(...all);
  const sx = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
  const sy = (v: number) => m.t + (1 - (v - yMin) / (yMax - yMin || 1)) * ih;

  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": spec.yLabel });
  for (const t of niceTicks(yMin, yMax, 4)) {
    svg.append(el("line", { x1: m.l, x2: width - m.r, y1: sy(t), y2: sy(t), class: "grid" }));
    const label = el("text", { x: m.l - 8, y: sy(t), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" });
    label.textContent = spec.yFormat(t);
    svg.append(label);
  }
  for (const t of niceTicks(x0, x1, Math.max(2, Math.floor(iw / 110)))) {
    const label = el("text", { x: sx(t), y: height - 6, class: "tick", "text-anchor": "middle" });
    label.textContent = spec.xFormat(t);
    svg.append(label);
  }
  if (yMin < 0 && yMax > 0) svg.append(el("line", { x1: m.l, x2: width - m.r, y1: sy(0), y2: sy(0), class: "zero" }));

  if (spec.lo && spec.hi) {
    const top = xs.map((x, i) => `${sx(x)},${sy(spec.hi![i])}`);
    const bottom = xs.map((x, i) => `${sx(x)},${sy(spec.lo![i])}`).reverse();
    svg.append(el("polygon", { points: [...top, ...bottom].join(" "), fill: spec.color, opacity: 0.14 }));
  }
  svg.append(el("polyline", {
    points: xs.map((x, i) => `${sx(x)},${sy(spec.y[i])}`).join(" "),
    fill: "none", stroke: spec.color, "stroke-width": 2, "stroke-linejoin": "round",
  }));

  (spec.marks ?? []).forEach((mark, i) => {
    const g = el("g", { class: `mark${mark.active ? " active" : ""}`, tabindex: 0, role: "button", "aria-label": mark.label });
    g.append(el("line", { x1: sx(mark.x), x2: sx(mark.x), y1: m.t, y2: m.t + ih }));
    g.append(el("rect", { x: sx(mark.x) - 6, y: m.t, width: 12, height: ih, fill: "transparent" }));
    if (spec.onMark) {
      g.addEventListener("click", () => spec.onMark?.(i));
      g.addEventListener("keydown", (e) => {
        if ((e as KeyboardEvent).key === "Enter") spec.onMark?.(i);
      });
    }
    svg.append(g);
  });

  const cross = el("line", { y1: m.t, y2: m.t + ih, class: "cross", visibility: "hidden" });
  const dot = el("circle", { r: 4, fill: spec.color, stroke: "var(--bg)", "stroke-width": 2, visibility: "hidden" });
  svg.append(cross, dot);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;

  svg.addEventListener("pointermove", (e) => {
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const xv = x0 + ((px - m.l) / iw) * (x1 - x0);
    let best = 0;
    for (let i = 1; i < xs.length; i += 1) if (Math.abs(xs[i] - xv) < Math.abs(xs[best] - xv)) best = i;
    const cx = sx(xs[best]);
    cross.setAttribute("x1", String(cx));
    cross.setAttribute("x2", String(cx));
    dot.setAttribute("cx", String(cx));
    dot.setAttribute("cy", String(sy(spec.y[best])));
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("visibility", "visible");
    const band = spec.lo && spec.hi ? `<span>p10–p90 ${spec.yFormat(spec.lo[best])} – ${spec.yFormat(spec.hi[best])}</span>` : "";
    tip.innerHTML = `<b>${spec.yFormat(spec.y[best])}</b><span>${spec.yLabel}</span>${band}<span>${spec.xFormat(xs[best])}</span>`;
    tip.hidden = false;
    tip.style.left = `${Math.min(cx + 12, width - 170)}px`;
    tip.style.top = `${m.t}px`;
  });
  svg.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  host.append(svg, tip);
}
