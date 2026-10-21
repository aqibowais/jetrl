import type { Curve, Manifest } from "./data";
import { lineChart } from "./ui/chart";

const $ = (id: string) => document.getElementById(id) as HTMLElement;
const steps = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${Math.round(v / 1e3)}k`);

export function renderReport(manifest: Manifest, curve: Curve, onPick: (id: string) => void): void {
  const best = manifest.checkpoints.find((c) => c.id === manifest.best) ?? manifest.checkpoints[manifest.checkpoints.length - 1];
  const hand = manifest.heuristic?.mean_lap?.toFixed(1) ?? "--";
  const totalSteps = curve.bins.length ? curve.bins[curve.bins.length - 1].timesteps : best.timesteps;

  $("lede").textContent =
    `Best specialist checkpoint, episode ${best.episode.toLocaleString()}: ${pctText(best.eval.lap_rate)} of laps from ${best.eval.episodes} starts` +
    (best.eval.mean_lap ? `, ${best.eval.mean_lap.toFixed(1)} s mean lap` : "") +
    `, against ${hand} s for the hand-written pilot in test_env.py. Training: ${steps(totalSteps)} steps, ${curve.episodes.toLocaleString()} episodes, CPU.`;

  const draw = () => {
    if (!curve.bins.length) {
      $("rewardChart").textContent = "No reward log exported.";
      return;
    }
    const x = curve.bins.map((b) => b.timesteps);
    const marks = manifest.checkpoints.map((c) => ({ x: c.timesteps, label: `Fly episode ${c.episode}`, active: c.id === manifest.best }));
    lineChart($("rewardChart"), {
      x,
      y: curve.bins.map((b) => b.reward),
      lo: curve.bins.map((b) => b.reward_lo),
      hi: curve.bins.map((b) => b.reward_hi),
      color: "#ffb000",
      xFormat: (v) => `${steps(v)} steps`,
      yFormat: (v) => v.toFixed(0),
      yLabel: "episode return",
      marks,
      onMark: (i) => onPick(manifest.checkpoints[i].id),
    });
    lineChart($("lapChart"), {
      x,
      y: curve.bins.map((b) => b.laps * 100),
      color: "#4cc38a",
      xFormat: (v) => `${steps(v)} steps`,
      yFormat: (v) => `${v.toFixed(0)}%`,
      yLabel: "episodes ending in a lap",
      yMin: 0,
      yMax: 100,
    });
  };
  draw();
  let t = 0;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = window.setTimeout(draw, 150);
  });

  $("evalCaption").textContent =
    `Deterministic actions, ${manifest.eval_seeds} seeds, full laps from the start line. Click a row to fly it.`;
  const head = `<thead><tr><th>Episode</th><th>Env steps</th><th>Lap rate</th><th>Mean lap</th><th>Gates</th><th>Return</th><th>Outcome mix</th></tr></thead>`;
  const rows = manifest.checkpoints.map((c) => {
    const e = c.eval;
    const mix = Object.entries(e.outcomes).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k.replace("_", " ")} ${n}`).join(", ");
    return `<tr data-id="${c.id}" tabindex="0" class="${c.id === manifest.best ? "best" : ""}">
      <td>${c.episode.toLocaleString()}</td><td>${steps(c.timesteps)}</td>
      <td><span class="cellbar"><i style="width:${e.lap_rate * 100}%"></i></span>${Math.round(e.lap_rate * 100)}%</td>
      <td>${e.mean_lap ? `${e.mean_lap.toFixed(2)} s` : "–"}</td><td>${e.mean_gates.toFixed(1)}</td>
      <td>${e.mean_reward.toFixed(1)}</td><td class="mix">${mix}</td></tr>`;
  }).join("");
  const table = $("evalTable");
  table.innerHTML = `${head}<tbody>${rows}</tbody>`;
  const pick = (e: Event) => {
    const row = (e.target as HTMLElement).closest<HTMLElement>("tr[data-id]");
    if (row?.dataset.id) onPick(row.dataset.id);
  };
  table.addEventListener("click", pick);
  table.addEventListener("keydown", (e) => {
    if (e.key === "Enter") pick(e);
  });
  renderStudies(manifest);
}

const pctText = (v: number) => `${Math.round(v * 100)}%`;

function bar(label: string, value: number, color: string, note = ""): string {
  return `<div class="hbar"><span class="hb-lbl">${label}</span><span class="hb-track"><i style="width:${Math.max(0.5, value * 100)}%;background:${color}"></i></span><b>${pctText(value)}</b><em>${note}</em></div>`;
}

/** Generalization and ablation sections. Missing data hides the section. */
export function renderStudies(manifest: Manifest): void {
  const best = manifest.checkpoints.find((c) => c.id === manifest.best);
  const g = manifest.generalist;
  const su = manifest.specialist_unseen;
  const general = document.getElementById("general");
  if (general) general.hidden = !(g && su && best);
  if (g && su && best) {
    $("generalCaption").textContent =
      `Lap completion on the fixed circuit and on ${g.unseen.tracks} generated test circuits. The generalist checkpoint was picked on a separate validation set.`;
    $("generalBars").innerHTML =
      `<div class="hgroup"><h4><i style="background:#ffb000"></i>Specialist, episode ${best.episode.toLocaleString()}</h4>` +
      bar("Fixed circuit", best.eval.lap_rate, "#ffb000", best.eval.mean_lap ? `${best.eval.mean_lap.toFixed(1)} s mean lap` : "") +
      bar("Unseen circuits", su.lap_rate, "#ffb000", `${pctText(su.gate_share)} of gates cleared`) +
      `</div><div class="hgroup"><h4><i style="background:#b392f0"></i>Generalist, episode ${g.episode.toLocaleString()}</h4>` +
      bar("Fixed circuit", g.fixed.lap_rate, "#b392f0", g.fixed.mean_lap ? `${g.fixed.mean_lap.toFixed(1)} s mean lap` : "") +
      bar("Unseen circuits", g.unseen.lap_rate, "#b392f0", `${pctText(g.unseen.gate_share)} of gates cleared`) +
      `</div>`;
    const slower = g.fixed.mean_lap && best.eval.mean_lap ? g.fixed.mean_lap - best.eval.mean_lap : null;
    $("generalNote").textContent =
      "The generalist trains on generated circuits with the gate-index input zeroed; on one fixed track that input acts as a lap position and invites memorizing the line." +
      (slower !== null && slower > 0 ? ` On the specialist's own circuit it is ${slower.toFixed(1)} s slower per lap.` : "");
  }

  const ablation = manifest.ablation ?? [];
  const sec = document.getElementById("ablation");
  if (sec) sec.hidden = !ablation.length;
  if (ablation.length) {
    // Bar length: env steps until the first checkpoint that lapped every start. Shorter is faster learning.
    const budget = 4e6;
    $("ablationBars").innerHTML = ablation
      .map((a) => {
        const reached = a.runs.filter((r) => r.lap_rate >= 1);
        const mean = reached.length ? reached.reduce((t, r) => t + r.timesteps, 0) / reached.length : budget;
        const dots = a.runs.map((r) => `<i class="dot" style="left:${(Math.min(r.timesteps, budget) / budget) * 100}%" title="seed ${r.seed}: ${steps(r.timesteps)}, ${pctText(r.lap_rate)} laps"></i>`).join("");
        const laps = a.runs.filter((r) => r.mean_lap).map((r) => r.mean_lap!.toFixed(1));
        return `<div class="hbar abl"><span class="hb-lbl">${a.label}<small>${a.detail}</small></span><span class="hb-track"><i style="width:${(mean / budget) * 100}%"></i>${dots}</span><b>${steps(mean)}</b><em>${reached.length}/${a.runs.length} reached 100%, laps ${laps.join(" / ")} s</em></div>`;
      })
      .join("");
    const full = ablation.find((a) => a.id === "full");
    if (full && full.runs.length > 1) {
      const t = full.runs.map((r) => r.timesteps);
      $("ablationNote").textContent =
        `Every configuration learned to lap every start within the budget. Across three seeds of the same setup, the first such checkpoint came between ${steps(Math.min(...t))} and ${steps(Math.max(...t))} steps; ` +
        "with one seed for the other configurations, only a difference well outside that spread would say anything about a fix.";
    }
  }
}
