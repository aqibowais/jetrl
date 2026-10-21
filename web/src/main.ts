import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-sans-condensed/400.css";
import "@fontsource/ibm-plex-sans-condensed/500.css";
import "@fontsource/ibm-plex-sans-condensed/600.css";
import "./style.css";

import { loadCurve, loadManifest, loadPolicy, loadTrack, type Manifest } from "./data";
import { Instruments } from "./hud/instruments";
import { Pilot } from "./input/pilot";
import { renderReport } from "./report";
import { HUD_GREEN, JetVisual, World, type CameraMode } from "./scene/world";
import { DT, JetSim, OBS_DIM, type EndReason, type StepResult, type Track } from "./sim/jet";
import type { Policy } from "./sim/policy";
import { mulberry32, randomTrack } from "./sim/tracks";
import { RewardLedger, decisions, narrate, reading } from "./ui/explain";
import { Minimap } from "./ui/minimap";
import { NetworkView } from "./ui/network";

type Mode = "story" | "watch" | "lab" | "evolve" | "race";

const SPEC = "#ffb000";
const GEN = "#b392f0";
const HUMAN = "#5cc8ff";
const hex = (c: string) => parseInt(c.slice(1), 16);
const fmtLap = (s: number | null | undefined) => (s == null ? "--.--" : s.toFixed(2));
const pct = (v: number) => `${Math.round(v * 100)}%`;

const REASON: Record<EndReason, string> = {
  "": "",
  crash_ground: "Hit the ground",
  crash_bounds: "Left the airspace",
  lap_complete: "Lap complete",
  stalled: "Stalled: 20 s without a gate",
  timeout: "Time limit",
};

const $ = <T extends HTMLElement = HTMLElement>(id: string) => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} missing`);
  return node as T;
};

/** One jet and whoever flies it. */
class Runner {
  readonly sim: JetSim;
  readonly obs = new Float64Array(OBS_DIM);
  readonly ledger = new RewardLedger();
  action = [0, 0, 0, 0];
  done = false;
  reason: EndReason = "";
  doneAt = 0;
  last: StepResult | null = null;

  constructor(
    track: Track,
    readonly visual: JetVisual,
    readonly policy: Policy | null,
    readonly name: string,
    readonly color: string,
  ) {
    this.sim = new JetSim(track);
  }

  reset(seed: number): void {
    this.sim.reset(mulberry32(seed));
    this.done = false;
    this.reason = "";
    this.last = null;
    this.ledger.reset();
    this.visual.reset();
    this.visual.commit(this.sim.state);
    this.sim.observation(this.obs, this.policy?.mask);
  }

  step(human?: number[]): void {
    if (this.done) return;
    if (this.policy) this.policy.act(this.obs, this.action);
    else if (human) this.action = human;
    const r = this.sim.step(this.action);
    this.last = r;
    this.ledger.record(this.sim.state.time, r, this.sim.state, this.sim.track.radius, DT);
    this.visual.commit(this.sim.state);
    this.sim.observation(this.obs, this.policy?.mask);
    if (r.terminated || r.truncated) {
      this.done = true;
      this.reason = r.reason;
      this.doneAt = performance.now();
    }
  }

  gatesCleared(): number {
    return this.sim.state.cleared.filter(Boolean).length;
  }
}

interface StoryStep {
  title: string;
  body: string;
  track: "fixed" | "unseen";
  pilots: { id: string; name: string; color: string }[];
  camera: CameraMode;
}

function buildStory(m: Manifest): StoryStep[] {
  const cps = m.checkpoints;
  const early = cps[0];
  const firstLap = cps.find((c) => c.eval.lap_rate >= 0.9) ?? cps[cps.length - 1];
  const best = cps.find((c) => c.id === m.best) ?? cps[cps.length - 1];
  const g = m.generalist;
  const su = m.specialist_unseen;
  const hand = m.heuristic?.mean_lap ?? null;
  const vsHand =
    hand && best.eval.mean_lap
      ? ` The hand-written pilot in test_env.py averages ${hand.toFixed(1)} s, so the policy is ${Math.abs(hand - best.eval.mean_lap).toFixed(1)} s ${best.eval.mean_lap < hand ? "faster" : "slower"}.`
      : "";
  const steps: StoryStep[] = [
    {
      title: `Episode ${early.episode.toLocaleString()}`,
      body: `Early in training. The green line points at the gate it is aiming for. It clears ${early.eval.mean_gates.toFixed(1)} of 9 gates on average before crashing or stalling.`,
      track: "fixed",
      pilots: [{ id: early.id, name: `Episode ${early.episode}`, color: SPEC }],
      camera: "chase",
    },
    {
      title: `Episode ${firstLap.episode.toLocaleString()}: first laps`,
      body: `${pct(firstLap.eval.lap_rate)} of evaluation starts end in a full lap. The panel on the right shows what it observes and what it outputs each frame.`,
      track: "fixed",
      pilots: [{ id: firstLap.id, name: `Episode ${firstLap.episode}`, color: SPEC }],
      camera: "chase",
    },
    {
      title: `Episode ${best.episode.toLocaleString()}: the specialist`,
      body: `${pct(best.eval.lap_rate)} of laps, ${best.eval.mean_lap ? best.eval.mean_lap.toFixed(1) : "--"} s on average.${vsHand}`,
      track: "fixed",
      pilots: [{ id: best.id, name: "Specialist", color: SPEC }],
      camera: "tv",
    },
  ];
  if (g && su) {
    steps.push(
      {
        title: "A circuit it has never seen",
        body: `On ${su.tracks} generated circuits the specialist finishes ${pct(su.lap_rate)} of laps and clears ${pct(su.gate_share)} of gates. It trained on one track only.`,
        track: "unseen",
        pilots: [{ id: best.id, name: "Specialist", color: SPEC }],
        camera: "chase",
      },
      {
        title: "The generalist",
        body: `Same network and algorithm, trained on generated circuits with the gate-index input hidden. ${pct(g.unseen.lap_rate)} of laps on ${g.unseen.tracks} held-out circuits.`,
        track: "unseen",
        pilots: [{ id: g.id, name: "Generalist", color: GEN }],
        camera: "chase",
      },
      {
        title: "Your turn",
        body: "New track generates circuits and flies both policies on them. Race lets you fly against the policy.",
        track: "unseen",
        pilots: [
          { id: best.id, name: "Specialist", color: SPEC },
          { id: g.id, name: "Generalist", color: GEN },
        ],
        camera: "overview",
      },
    );
  }
  return steps;
}

async function boot(): Promise<void> {
  const [fixedTrack, manifest, curve] = await Promise.all([loadTrack(), loadManifest(), loadCurve()]);
  const world = new World($<HTMLCanvasElement>("view"), fixedTrack);
  const hud = new Instruments($<HTMLCanvasElement>("hud"));
  const net = new NetworkView($<HTMLCanvasElement>("net"));
  const minimap = new Minimap($<HTMLCanvasElement>("minimap"));
  const pilot = new Pilot();
  const byId = new Map(manifest.checkpoints.map((c) => [c.id, c]));
  const story = buildStory(manifest);
  const storySeed = manifest.story_seed ?? 101;
  const generalist = manifest.generalist;

  let mode: Mode = "story";
  let speed = 1;
  let track: Track = fixedTrack;
  let runners: Runner[] = [];
  let focus: Runner | null = null;
  let jitter = 1;
  let acc = 0;
  let session = 0;
  let storyIndex = 0;
  let labSeed = 0;
  const tally = { runs: 0, spec: 0, gen: 0 };
  let countdown = 0;
  let raceLive = false;
  let selected = manifest.best;
  let roundRecorded = false;
  let clock = 0;

  const resize = () => {
    world.resize();
    hud.resize();
    net.resize();
    minimap.resize();
  };
  window.addEventListener("resize", resize);

  function useTrack(t: Track): void {
    track = t;
    world.setTrack(t);
    minimap.setTrack(t);
  }

  function clearRunners(): void {
    runners.forEach((r) => r.visual.dispose());
    runners = [];
    focus = null;
  }

  async function spawn(pilots: { id: string | null; name: string; color: string; opacity?: number }[], seed: number): Promise<Runner[] | null> {
    const token = ++session;
    const policies = await Promise.all(pilots.map((p) => (p.id ? loadPolicy(p.id) : Promise.resolve(null))));
    if (token !== session) return null;
    clearRunners();
    runners = pilots.map((p, i) => {
      const r = new Runner(track, new JetVisual(world.root, hex(p.color), p.opacity ?? 1), policies[i], p.name, p.color);
      r.reset(seed);
      return r;
    });
    roundRecorded = false;
    return runners;
  }

  function setFocus(r: Runner | null): void {
    focus = r;
    world.target = r?.visual ?? null;
    const who = $("who");
    who.querySelector("span")!.textContent = r ? r.name : "--";
    (who.querySelector("i") as HTMLElement).style.background = r?.color ?? "transparent";
    $("seesTag").textContent = r?.name ?? "--";
  }

  function banner(text: string, tone: "good" | "bad" | "info" = "info"): void {
    const b = $("banner");
    b.textContent = text;
    b.dataset.tone = tone;
    b.hidden = !text;
  }

  function setCamera(c: CameraMode): void {
    world.setMode(c);
    document.querySelectorAll("[data-cam]").forEach((x) => x.classList.toggle("on", x.getAttribute("data-cam") === c));
  }


  async function showStory(i: number): Promise<void> {
    storyIndex = Math.max(0, Math.min(story.length - 1, i));
    const step = story[storyIndex];
    useTrack(step.track === "fixed" ? fixedTrack : randomTrack(storySeed));
    const made = await spawn(step.pilots, 7);
    if (!made) return;
    setFocus(made[made.length - 1]);
    setCamera(step.camera);
    $("storyStep").textContent = `${storyIndex + 1} / ${story.length}`;
    $("storyTitle").textContent = step.title;
    $("storyBody").textContent = step.body;
    $("storyDots").innerHTML = story.map((_, k) => `<i class="${k === storyIndex ? "on" : k < storyIndex ? "seen" : ""}"></i>`).join("");
    ($("storyPrev") as HTMLButtonElement).disabled = storyIndex === 0;
    $("storyNext").textContent = storyIndex === story.length - 1 ? (generalist ? "Open New track" : "Watch") : "Next";
    banner("");
  }

  async function startWatch(id: string): Promise<void> {
    selected = id;
    const isGen = !!generalist && id === generalist.id;
    useTrack(fixedTrack);
    const made = await spawn([{ id, name: isGen ? "Generalist" : `Episode ${byId.get(id)?.episode.toLocaleString() ?? id}`, color: isGen ? GEN : SPEC }], jitter++);
    if (!made) return;
    setFocus(made[0]);
    document.querySelectorAll<HTMLButtonElement>("#ticks button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.id === id)));
    banner("");
  }

  async function startLab(newSeed = true): Promise<void> {
    if (!generalist) return;
    if (newSeed || !labSeed) labSeed = 1000 + Math.floor(Math.random() * 900000);
    useTrack(randomTrack(labSeed));
    const made = await spawn(
      [
        { id: manifest.best, name: "Specialist", color: SPEC },
        { id: generalist.id, name: "Generalist", color: GEN },
      ],
      labSeed,
    );
    if (!made) return;
    setFocus(made[1]);
    $("labSeed").textContent = `Seed ${labSeed}, ${track.centers.length} gates, ${(track.lapLength / 1000).toFixed(2)} km`;
    banner("");
  }

  async function startEvolution(): Promise<void> {
    useTrack(fixedTrack);
    const n = manifest.checkpoints.length;
    const made = await spawn(
      manifest.checkpoints.map((c, i) => {
        const t = n === 1 ? 1 : i / (n - 1);
        const mix = (a: number, b: number) => Math.round(a + (b - a) * (0.15 + 0.85 * t));
        const color = `#${[mix(0x55, 0xff), mix(0x60, 0xb0), mix(0x6a, 0x00)].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
        return { id: c.id, name: `ep ${c.episode.toLocaleString()}`, color, opacity: 0.4 + 0.6 * t };
      }),
      7,
    );
    if (!made) return;
    setFocus(made.find((r) => r.policy?.id === manifest.best) ?? made[made.length - 1]);
    setCamera("overview");
  }

  async function startRace(): Promise<void> {
    // Start stays disabled until the ghost's policy is loaded, so an early click can't be lost.
    $<HTMLButtonElement>("raceGo").disabled = true;
    useTrack(fixedTrack);
    const made = await spawn(
      [
        { id: manifest.best, name: "Policy", color: SPEC, opacity: 0.55 },
        { id: null, name: "You", color: HUMAN },
      ],
      11,
    );
    if (!made) return;
    setFocus(made[1]);
    pilot.reset();
    raceLive = false;
    countdown = 0;
    $("raceCard").hidden = false;
    $("result").hidden = true;
    $<HTMLButtonElement>("raceGo").disabled = false;
    renderPb();
  }

  function setMode(next: Mode): void {
    mode = next;
    document.body.dataset.mode = next;
    document.querySelectorAll<HTMLButtonElement>("button[data-mode]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === next)));
    $("story").hidden = next !== "story";
    $("lab").hidden = next !== "lab";
    $("legend").hidden = next !== "evolve";
    $("raceCard").hidden = true;
    $("result").hidden = true;
    pilot.active = next === "race";
    if (next === "story") void showStory(storyIndex);
    else if (next === "watch") {
      setCamera("chase");
      void startWatch(selected);
    } else if (next === "lab") {
      setCamera("chase");
      void startLab(!labSeed);
    } else if (next === "evolve") void startEvolution();
    else {
      setCamera("chase");
      void startRace();
    }
  }


  document.querySelectorAll<HTMLButtonElement>("button[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode as Mode)));
  document.querySelectorAll<HTMLButtonElement>("[data-cam]").forEach((b) => b.addEventListener("click", () => setCamera(b.dataset.cam as CameraMode)));
  document.querySelectorAll<HTMLButtonElement>("[data-speed]").forEach((b) =>
    b.addEventListener("click", () => {
      speed = Number(b.dataset.speed);
      document.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("on", x === b));
    }),
  );
  $("storyPrev").addEventListener("click", () => void showStory(storyIndex - 1));
  $("storyReplay").addEventListener("click", () => void showStory(storyIndex));
  $("storyNext").addEventListener("click", () => {
    if (storyIndex < story.length - 1) void showStory(storyIndex + 1);
    else setMode(generalist ? "lab" : "watch");
  });
  $("labNew").addEventListener("click", () => void startLab(true));
  $("labPilots").addEventListener("click", (e) => {
    const card = (e.target as HTMLElement).closest<HTMLElement>("[data-runner]");
    if (card) setFocus(runners[Number(card.dataset.runner)] ?? focus);
  });
  $("legend").addEventListener("click", (e) => {
    const row = (e.target as HTMLElement).closest<HTMLElement>("[data-runner]");
    if (row) setFocus(runners[Number(row.dataset.runner)] ?? focus);
  });
  $("panelToggle").addEventListener("click", () => {
    document.body.classList.toggle("panel-off");
    requestAnimationFrame(resize);
  });
  $<HTMLInputElement>("assist").addEventListener("change", (e) => {
    pilot.assist = (e.target as HTMLInputElement).checked;
  });
  $("raceGo").addEventListener("click", () => {
    $("raceCard").hidden = true;
    countdown = 3;
  });
  const restartRace = () =>
    void startRace().then(() => {
      $("raceCard").hidden = true;
      countdown = 3;
    });
  $("resAgain").addEventListener("click", restartRace);
  window.addEventListener("keydown", (e) => {
    if (mode === "race" && e.code === "KeyR") restartRace();
  });

  const tickItems: { id: string; label: string; rate: number; gen?: boolean }[] = manifest.checkpoints.map((c) => ({
    id: c.id,
    label: c.episode >= 1000 ? `${(c.episode / 1000).toFixed(1)}k` : String(c.episode),
    rate: c.eval.lap_rate,
  }));
  if (generalist) tickItems.push({ id: generalist.id, label: "GEN", rate: generalist.fixed.lap_rate, gen: true });
  $("ticks").innerHTML = tickItems
    .map((t) => `<button role="option" data-id="${t.id}" aria-selected="false" class="${t.gen ? "gen" : ""}" title="${Math.round(t.rate * 100)}% laps on the fixed circuit">
      <span class="ep">${t.label}</span><span class="meter"><i style="width:${Math.max(2, t.rate * 100)}%"></i></span><span class="pct">${Math.round(t.rate * 100)}%</span>
    </button>`)
    .join("");
  $("ticks").addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-id]");
    if (!b?.dataset.id) return;
    if (mode !== "watch") {
      selected = b.dataset.id;
      setMode("watch");
    } else void startWatch(b.dataset.id);
  });

  renderReport(manifest, curve, (id) => {
    window.scrollTo({ top: 0, behavior: "smooth" });
    selected = id;
    setMode("watch");
  });


  function readPb(): number | null {
    try {
      const v = Number(localStorage.getItem("jetrl.pb"));
      return v > 0 ? v : null;
    } catch {
      return null;
    }
  }
  function renderPb(): void {
    const pb = readPb();
    const ai = byId.get(manifest.best)?.eval.mean_lap;
    $("pbRead").textContent = `Your best ${pb ? `${pb.toFixed(2)} s` : "--"}, policy ${ai ? `${ai.toFixed(2)} s` : "--"}`;
  }
  function finishRace(): void {
    const [ghost, you] = runners;
    raceLive = false;
    const won = you.reason === "lap_complete";
    const yourLap = won ? you.sim.state.lapTime : null;
    const aiLap = ghost.reason === "lap_complete" ? ghost.sim.state.lapTime : null;
    $("resTitle").textContent = won ? "Lap complete" : REASON[you.reason];
    $("resYou").textContent = yourLap ? `${yourLap.toFixed(2)} s` : "DNF";
    $("resAi").textContent = aiLap ? `${aiLap.toFixed(2)} s` : ghost.done ? "DNF" : "flying";
    let note = "";
    if (yourLap && aiLap) note = yourLap < aiLap ? `You beat the policy by ${(aiLap - yourLap).toFixed(2)} s.` : `The policy was ${(yourLap - aiLap).toFixed(2)} s faster.`;
    else if (!won) note = `You cleared ${you.gatesCleared()} of ${track.centers.length} gates.`;
    if (yourLap) {
      const pb = readPb();
      if (!pb || yourLap < pb) {
        try {
          localStorage.setItem("jetrl.pb", String(yourLap));
        } catch {
          /* storage blocked */
        }
        note += " New personal best.";
      }
    }
    $("resNote").textContent = note;
    $("result").hidden = false;
  }


  function renderPanels(): void {
    const r = focus;
    if (!r) return;
    const s = r.sim.state;
    const read = reading(r.obs, s);
    $("seeGate").textContent = `${read.gate + 1} of ${track.centers.length}`;
    $("seeDist").textContent = `${Math.round(read.distance)} m`;
    $("seeBearing").textContent = Math.abs(read.bearing) < 3 ? "straight ahead" : `${Math.abs(Math.round(read.bearing))}° ${read.bearing > 0 ? "right" : "left"}`;
    $("seeHeight").textContent = Math.abs(read.height) < 3 ? "level" : `${Math.abs(Math.round(read.height))} m ${read.height > 0 ? "above" : "below"}`;
    $("seeSpeed").textContent = `${Math.round(s.speed * 1.944)} kt`;
    $("bearingArrow").style.transform = `rotate(${Math.max(-90, Math.min(90, read.bearing))}deg)`;
    $("narrate").textContent = r.policy ? narrate(read, r.action) : "You are flying. The panel shows what the policy would see.";
    $("decide").innerHTML = decisions(r.action)
      .map((d) => {
        const w = Math.abs(d.value) * 50;
        return `<div class="dec"><span class="lbl">${d.label}</span><span class="track"><i class="${d.value < 0 ? "neg" : "pos"}" style="${d.value < 0 ? "right:50%" : "left:50%"};width:${w}%"></i></span><span class="txt">${d.text}</span></div>`;
      })
      .join("");
    $("rewardTag").textContent = `${s.episodeReward >= 0 ? "+" : ""}${s.episodeReward.toFixed(1)} total`;
    $("ledger").innerHTML =
      r.ledger.events.map((e) => `<li class="${e.tone}"><b>${e.value >= 0 ? "+" : ""}${e.value.toFixed(1)}</b><span>${e.text}</span><em>${e.t.toFixed(1)} s</em></li>`).join("") ||
      `<li class="empty">Rewards appear here as the jet earns them.</li>`;

    if (mode === "lab") {
      $("labPilots").innerHTML = runners
        .map((x, i) => {
          const st = x.done ? (x.reason === "lap_complete" ? `${fmtLap(x.sim.state.lapTime)} s` : REASON[x.reason]) : "flying";
          return `<button type="button" class="pilot ${x === focus ? "on" : ""} ${x.done ? (x.reason === "lap_complete" ? "ok" : "bad") : ""}" data-runner="${i}" style="--c:${x.color}">
            <i></i><span class="nm">${x.name}</span><span class="g">${x.gatesCleared()}/${track.centers.length} gates</span><b>${st}</b></button>`;
        })
        .join("");
      $("labTally").textContent = tally.runs
        ? `So far: generalist ${tally.gen}/${tally.runs}, specialist ${tally.spec}/${tally.runs}`
        : "Click a pilot to follow it. Both start from the same spot.";
    }
    if (mode === "evolve") {
      $("legend").innerHTML =
        `<h4>Every checkpoint, same start</h4>` +
        runners
          .map((x, i) => {
            const st = x.done ? (x.reason === "lap_complete" ? `${fmtLap(x.sim.state.lapTime)} s` : "out") : `${x.gatesCleared()}/${track.centers.length}`;
            return `<div class="lg-row${x === focus ? " on" : ""}${x.done && x.reason !== "lap_complete" ? " dead" : ""}" data-runner="${i}"><i style="background:${x.color}"></i><span>${x.name}</span><b>${st}</b></div>`;
          })
          .join("");
    }
  }


  let last = performance.now();
  let panelTimer = 0;
  function frame(now: number): void {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    clock += dt;

    if (mode === "race" && countdown > 0) {
      countdown -= dt;
      banner(countdown > 0 ? String(Math.ceil(countdown)) : "GO");
      if (countdown <= 0) {
        raceLive = true;
        setTimeout(() => mode === "race" && raceLive && banner(""), 600);
      }
    }
    const running = mode !== "race" || raceLive;
    if (running && runners.length) {
      acc += dt * (mode === "race" ? 1 : speed);
      let guard = 0;
      while (acc >= DT && guard < 16) {
        acc -= DT;
        guard += 1;
        for (const r of runners) r.step(r.policy ? undefined : pilot.command(r.sim.state, DT));
      }
      if (guard === 16) acc = 0;
    }
    const alpha = Math.min(1, acc / DT);
    runners.forEach((r) => r.visual.draw(alpha, clock));

    if ((mode === "story" || mode === "watch") && focus?.done && runners.every((r) => r.done)) {
      banner(focus.reason === "lap_complete" ? `Lap ${fmtLap(focus.sim.state.lapTime)} s` : REASON[focus.reason], focus.reason === "lap_complete" ? "good" : "bad");
      if (now - Math.max(...runners.map((r) => r.doneAt)) > 2600) {
        const seed = mode === "story" ? 7 : jitter++;
        runners.forEach((r) => r.reset(seed));
        banner("");
      }
    } else if (mode === "lab" && runners.length && runners.every((r) => r.done)) {
      if (!roundRecorded) {
        roundRecorded = true;
        tally.runs += 1;
        if (runners[0].reason === "lap_complete") tally.spec += 1;
        if (runners[1]?.reason === "lap_complete") tally.gen += 1;
      }
      banner("Next track");
      if (now - Math.max(...runners.map((r) => r.doneAt)) > 3500) void startLab(true);
    } else if (mode === "evolve" && runners.length && runners.every((r) => r.done)) {
      if (now - Math.max(...runners.map((r) => r.doneAt)) > 2500) runners.forEach((r) => r.reset(7));
    } else if (mode === "race" && raceLive && runners[1]?.done) {
      finishRace();
    } else if (mode === "lab" || mode === "evolve") {
      banner("");
    }

    if (focus) {
      const s = focus.sim.state;
      world.setGates(s.nextGate, s.cleared, focus.done ? null : s.position);
      const read = reading(focus.obs, s);
      hud.inset = !document.body.classList.contains("panel-off") && window.innerWidth > 860 ? 340 : 0;
      hud.draw(s, HUD_GREEN, world.mode === "overview" || world.mode === "orbit" || world.mode === "tv" || window.innerWidth < 760, {
        gate: s.nextGate,
        gates: track.centers.length,
        gateDistance: read.distance,
      });
      $("gateRead").textContent = `${String(focus.gatesCleared()).padStart(2, "0")}/${String(track.centers.length).padStart(2, "0")}`;
      $("lapRead").textContent = fmtLap(focus.sim.liveLapTime());
      $("retRead").textContent = s.episodeReward.toFixed(1);
      if (focus.policy) net.draw(focus.obs, focus.policy.hidden, focus.action);
      else net.draw(focus.obs, [], focus.action);
      minimap.draw(track, s.nextGate, s.cleared, runners.map((r) => ({ position: r.sim.state.position, color: r.color, focus: r === focus })));
    } else hud.clear();
    panelTimer -= dt;
    if (panelTimer <= 0) {
      panelTimer = 0.1;
      renderPanels();
    }
    world.render(dt, focus && !focus.done ? focus.sim.state.nextGate : null);
    requestAnimationFrame(frame);
  }

  resize();
  setMode("story");
  requestAnimationFrame(frame);
}

boot().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("afterbegin", `<div class="fatal">Could not load the flight data: ${String(err.message ?? err)}</div>`);
});
