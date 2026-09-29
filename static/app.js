import * as THREE from "three";

const DT_SMOOTH = 1 / 30;
const view = document.getElementById("view");
const horizon = document.getElementById("horizon");
const hctx = horizon.getContext("2d");
const rewardCanvas = document.getElementById("reward");
const rctx = rewardCanvas.getContext("2d");
const timeline = document.getElementById("timeline");
const tctx = timeline.getContext("2d");

const ui = {
  heading: document.getElementById("heading"),
  mode: document.getElementById("mode"),
  spd: document.getElementById("spd"),
  alt: document.getElementById("alt"),
  gforce: document.getElementById("gforce"),
  thr: document.getElementById("thr"),
  thrFill: document.getElementById("thrFill"),
  gate: document.getElementById("gate"),
  gateOf: document.getElementById("gateOf"),
  lap: document.getElementById("lap"),
  warn: document.getElementById("warn"),
  bestManual: document.getElementById("bestManual"),
  bestAi: document.getElementById("bestAi"),
  keys: document.getElementById("keys"),
  plan: document.getElementById("plan"),
  confirm: document.getElementById("confirm"),
  compareRead: document.getElementById("compareRead"),
  stop: document.getElementById("stop"),
  fly: document.getElementById("fly"),
  chase: document.getElementById("chase"),
  overview: document.getElementById("overview"),
  compare: document.getElementById("compare"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x8ebfe0);
scene.fog = new THREE.Fog(0xc5d7e6, 900, 3200);
const camera = new THREE.PerspectiveCamera(58, 1, 0.2, 6000);
camera.position.set(0, 18, -40);
const renderer = new THREE.WebGLRenderer({ canvas: view, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const sky = new THREE.Mesh(
  new THREE.SphereGeometry(4200, 28, 16),
  new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    vertexShader: "varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "varying vec3 vP; void main(){ float h = normalize(vP).y; vec3 top = vec3(0.36, 0.62, 0.86); vec3 mid = vec3(0.62, 0.80, 0.93); vec3 hor = vec3(0.86, 0.91, 0.93); float t = smoothstep(-0.02, 0.55, h); vec3 col = mix(hor, mix(mid, top, smoothstep(0.05, 0.7, h)), t); gl_FragColor = vec4(col, 1.0); }",
  })
);
scene.add(sky);

scene.add(new THREE.HemisphereLight(0xd7ecff, 0x6d8a62, 0.85));
const sun = new THREE.DirectionalLight(0xfff4e0, 1.55);
sun.position.set(240, 380, 120);
scene.add(sun);

const terrain = document.createElement("canvas");
terrain.width = 512;
terrain.height = 512;
const tg = terrain.getContext("2d");
tg.fillStyle = "#6e875c";
tg.fillRect(0, 0, 512, 512);
for (let i = 0; i < 2400; i += 1) {
  const shade = 70 + Math.floor(Math.random() * 55);
  tg.fillStyle = `rgba(${shade}, ${shade + 24}, ${shade - 8}, 0.35)`;
  tg.fillRect(Math.random() * 512, Math.random() * 512, 2 + Math.random() * 4, 2);
}
const groundMap = new THREE.CanvasTexture(terrain);
groundMap.wrapS = THREE.RepeatWrapping;
groundMap.wrapT = THREE.RepeatWrapping;
groundMap.repeat.set(24, 24);
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(3400, 80),
  new THREE.MeshStandardMaterial({ map: groundMap, color: 0xffffff, roughness: 1 })
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

for (let i = 0; i < 16; i += 1) {
  const height = 160 + (i % 5) * 55;
  const radius = 70 + (i % 4) * 28;
  const peak = new THREE.Mesh(
    new THREE.ConeGeometry(radius, height, 6),
    new THREE.MeshStandardMaterial({ color: i % 2 ? 0x9aa4a8 : 0xb7c0b4, roughness: 1 })
  );
  const angle = (i / 16) * Math.PI * 2;
  peak.position.set(Math.cos(angle) * 1900, height * 0.42, Math.sin(angle) * 1900);
  scene.add(peak);
}
for (let i = 0; i < 7; i += 1) {
  const cloud = new THREE.Mesh(
    new THREE.SphereGeometry(46 + (i % 3) * 16, 8, 6),
    new THREE.MeshBasicMaterial({ color: 0xf5f8fb })
  );
  cloud.scale.set(3.4, 0.32, 1.5);
  cloud.position.set(-1100 + i * 340, 620 + (i % 3) * 90, -900 + (i % 4) * 420);
  scene.add(cloud);
}

const jet = new THREE.Group();
const hullMat = new THREE.MeshStandardMaterial({ color: 0xb4c0c8, metalness: 0.35, roughness: 0.42 });
const darkMat = new THREE.MeshStandardMaterial({ color: 0x242a30, metalness: 0.2, roughness: 0.6 });
const body = new THREE.Mesh(new THREE.CylinderGeometry(0.46, 0.7, 6.6, 16), hullMat);
body.rotation.x = Math.PI / 2;
body.scale.set(1.25, 1, 0.62);
body.position.z = -0.2;
jet.add(body);
const nose = new THREE.Mesh(new THREE.ConeGeometry(0.46, 3.4, 16), hullMat);
nose.rotation.x = Math.PI / 2;
nose.scale.set(1.15, 1, 0.7);
nose.position.z = 4.15;
jet.add(nose);
const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.34, 0.7, 12), darkMat);
nozzle.rotation.x = Math.PI / 2;
nozzle.position.z = -3.7;
jet.add(nozzle);
[-1, 1].forEach((side) => {
  const wing = new THREE.Mesh(new THREE.BoxGeometry(5.4, 0.12, 1.7), hullMat);
  wing.position.set(side * 3.15, -0.05, -0.7);
  wing.rotation.y = side * -0.42;
  jet.add(wing);
});
const stab = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.1, 0.85), hullMat);
stab.position.set(0, 0.12, -3.7);
jet.add(stab);
[-1, 1].forEach((side) => {
  const fin = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1.45, 1.2), hullMat);
  fin.position.set(side * 0.72, 0.85, -3.55);
  fin.rotation.z = side * -0.22;
  jet.add(fin);
});
const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.46, 12, 8), darkMat);
canopy.scale.set(0.72, 0.5, 1.35);
canopy.position.set(0, 0.52, 1.15);
jet.add(canopy);
scene.add(jet);

const gateMeshes = [];
let centerline = null;
let gateCount = 9;
let trackStart = null;

const rewards = [];
const checkpoints = [];
let best = { manual: null, ai: null, ai_checkpoint: null };
let lastState = null;
let camMode = "chase";
let cameraSnap = 1;
let playback = 1;
let flyIntent = false;
let flyPending = false;
let compareArmed = false;
let compareA = null;
let compareB = null;
let selected = null;
let ws = null;
const keys = new Set();
let smoothedCam = new THREE.Vector3(-30, 24, -80);
const lookAt = new THREE.Vector3();

function bodyAxes(yaw, pitch, roll) {
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const forward = new THREE.Vector3(sy * cp, sp, cy * cp);
  const right0 = new THREE.Vector3(cy, 0, -sy);
  const up0 = new THREE.Vector3().crossVectors(forward, right0);
  const right = right0.clone().multiplyScalar(cr).addScaledVector(up0, -sr);
  const up = up0.clone().multiplyScalar(cr).addScaledVector(right0, sr);
  return { forward, right, up };
}

function alignNormal(normal) {
  const z = new THREE.Vector3(0, 0, 1);
  const n = normal.clone().normalize();
  if (n.dot(z) < -0.999) {
    return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
  }
  return new THREE.Quaternion().setFromUnitVectors(z, n);
}

function buildTrack(track) {
  gateCount = track.gates.length;
  trackStart = track.start;
  ui.gateOf.textContent = `/${String(gateCount).padStart(2, "0")}`;
  const geometry = new THREE.TorusGeometry(track.gates[0].radius, 1.7, 12, 64);
  for (const gate of track.gates) {
    const material = new THREE.MeshStandardMaterial({
      color: 0xd7efe4,
      emissive: 0x2f8f55,
      emissiveIntensity: 0.35,
      metalness: 0.05,
      roughness: 0.35,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(gate.center[0], gate.center[1], gate.center[2]);
    mesh.quaternion.copy(alignNormal(new THREE.Vector3(...gate.normal)));
    scene.add(mesh);
    gateMeshes.push(mesh);
  }
  const points = track.centerline.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  if (points.length) points.push(points[0].clone());
  centerline = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0xdff5c8 })
  );
  scene.add(centerline);
  addRunway(track.start);
  placeJet(track.start.position, track.start.yaw, track.start.pitch, 0);
}

function addRunway(start) {
  const deck = new THREE.Group();
  const asphalt = new THREE.Mesh(
    new THREE.PlaneGeometry(36, 520),
    new THREE.MeshStandardMaterial({ color: 0x5c656c, roughness: 0.95 })
  );
  asphalt.rotation.x = -Math.PI / 2;
  asphalt.position.y = 0.15;
  deck.add(asphalt);
  const paint = new THREE.Mesh(
    new THREE.PlaneGeometry(0.7, 460),
    new THREE.MeshBasicMaterial({ color: 0xf4f7f4 })
  );
  paint.rotation.x = -Math.PI / 2;
  paint.position.y = 0.22;
  deck.add(paint);
  for (let i = -4; i <= 4; i += 1) {
    const dash = new THREE.Mesh(
      new THREE.PlaneGeometry(0.55, 14),
      new THREE.MeshBasicMaterial({ color: 0xf4f7f4 })
    );
    dash.rotation.x = -Math.PI / 2;
    dash.position.set(i === 0 ? 8 : (i < 0 ? -8 : 8), 0.23, i * 48);
    deck.add(dash);
  }
  deck.position.set(start.position[0], 0, start.position[2]);
  deck.rotation.y = start.yaw;
  scene.add(deck);
}

function placeJet(position, yaw, pitch, roll) {
  const axes = bodyAxes(yaw, pitch, roll);
  const basis = new THREE.Matrix4().makeBasis(axes.right, axes.up, axes.forward);
  jet.position.set(position[0], position[1], position[2]);
  jet.quaternion.setFromRotationMatrix(basis);
  return axes;
}

function paintGates(cleared, nextIndex) {
  gateMeshes.forEach((mesh, index) => {
    const mat = mesh.material;
    if (cleared && cleared[index]) {
      mat.emissive.setHex(0x37ff8a);
      mat.emissiveIntensity = 1.6;
      mat.color.setHex(0xb8ffd0);
    } else if (index === nextIndex) {
      mat.emissive.setHex(0x39f0ff);
      mat.emissiveIntensity = 1.8;
      mat.color.setHex(0xe8fffb);
    } else {
      mat.emissive.setHex(0x2f8f55);
      mat.emissiveIntensity = 0.28;
      mat.color.setHex(0xd7efe4);
    }
  });
}

function formatTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "--:--.-";
  const sign = seconds < 0 ? "-" : "";
  const value = Math.abs(seconds);
  const minutes = Math.floor(value / 60);
  const secs = value - minutes * 60;
  return `${sign}${String(minutes).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
}

function formatHeading(yaw) {
  let deg = (yaw * 180) / Math.PI;
  deg = ((deg % 360) + 360) % 360;
  return `HDG ${String(Math.round(deg)).padStart(3, "0")}`;
}

function updateHud(state) {
  const alt = state.altitude;
  const low = alt < 18 || state.reason === "crash_ground" || state.reason === "crash_bounds";
  ui.spd.textContent = String(Math.round(state.speed)).padStart(3, "0");
  ui.alt.textContent = String(Math.max(0, Math.round(alt))).padStart(3, "0");
  ui.alt.classList.toggle("warn-num", low);
  ui.gforce.textContent = state.g_force.toFixed(1);
  ui.gforce.classList.toggle("warn-num", state.g_force > 3.2);
  ui.thr.textContent = `THR ${String(Math.round(state.throttle * 100)).padStart(3, "0")}`;
  ui.thrFill.style.height = `${Math.round(state.throttle * 100)}%`;
  ui.heading.textContent = formatHeading(state.yaw);
  const gateNo = (state.gate_index % gateCount) + 1;
  ui.gate.textContent = String(gateNo).padStart(2, "0");
  ui.lap.textContent = formatTime(state.lap_running || state.reason === "lap_complete" ? state.lap_time : state.lap_time);
  const modeNames = { idle: "IDLE", live: "LIVE", replay: "REPLAY", manual: "MANUAL" };
  let label = modeNames[state.mode] || "IDLE";
  if (state.mode === "replay" && state.checkpoint) label = `REPLAY ${state.checkpoint.replace("ep_", "")}`;
  if (state.mode === "replay") label += `  ${state.playback || playback}x`;
  ui.mode.textContent = label;
  if (state.reason === "crash_ground") ui.warn.textContent = "CRASH";
  else if (state.reason === "crash_bounds") ui.warn.textContent = "OUT OF RANGE";
  else if (alt < 18) ui.warn.textContent = "TERRAIN";
  else if (state.g_force > 3.2) ui.warn.textContent = "HIGH G";
  else ui.warn.textContent = "";
  paintGates(state.cleared, state.gate_index);
  const axes = placeJet(state.position, state.yaw, state.pitch, state.roll);
  return axes;
}

function updateBest(next) {
  if (!next) return;
  best = next;
  ui.bestManual.textContent = formatTime(best.manual);
  const ai = formatTime(best.ai);
  ui.bestAi.textContent = best.ai_checkpoint ? `${ai}  ${best.ai_checkpoint}` : ai;
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
  horizon.width = width * (window.devicePixelRatio || 1);
  horizon.height = height * (window.devicePixelRatio || 1);
  horizon.style.width = `${width}px`;
  horizon.style.height = `${height}px`;
  fitCanvas(rewardCanvas);
  fitCanvas(timeline);
  drawReward();
  drawTimeline();
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const cssWidth = Math.max(40, rect.width);
  const cssHeight = Math.max(36, rect.height);
  canvas.width = Math.floor(cssWidth * ratio);
  canvas.height = Math.floor(cssHeight * ratio);
}

function drawReward() {
  const w = rewardCanvas.width;
  const h = rewardCanvas.height;
  const ratio = window.devicePixelRatio || 1;
  rctx.setTransform(1, 0, 0, 1, 0, 0);
  rctx.clearRect(0, 0, w, h);
  rctx.fillStyle = "#7dff6a";
  rctx.font = `${11 * ratio}px Consolas, monospace`;
  rctx.fillText("REWARD", 8 * ratio, 14 * ratio);
  if (rewards.length < 2) return;
  const values = rewards.map((row) => row.reward);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (Math.abs(max - min) < 1) {
    min -= 1;
    max += 1;
  }
  const left = 8 * ratio;
  const right = w - 8 * ratio;
  const top = 20 * ratio;
  const bottom = h - 8 * ratio;
  const xOf = (i) => left + (i / (rewards.length - 1)) * (right - left);
  const yOf = (v) => bottom - ((v - min) / (max - min)) * (bottom - top);
  rctx.beginPath();
  values.forEach((value, i) => {
    const x = xOf(i);
    const y = yOf(value);
    if (i === 0) rctx.moveTo(x, y);
    else rctx.lineTo(x, y);
  });
  rctx.strokeStyle = "#7dff6a";
  rctx.lineWidth = ratio;
  rctx.stroke();
  rctx.fillStyle = "#f3fff2";
  rewards.forEach((row, i) => {
    if (!row.checkpoint) return;
    rctx.fillRect(xOf(i) - 2 * ratio, yOf(row.reward) - 2 * ratio, 4 * ratio, 4 * ratio);
  });
}

function episodeSpan() {
  const episodes = rewards.map((row) => row.episode).concat(checkpoints.map((row) => row.episode));
  if (!episodes.length) return [0, 1];
  return [Math.min(...episodes), Math.max(...episodes)];
}

function timelineX(episode, width) {
  const [minEp, maxEp] = episodeSpan();
  const span = Math.max(1, maxEp - minEp);
  const left = 16;
  const right = width - 16;
  return left + ((episode - minEp) / span) * (right - left);
}

function drawTimeline() {
  const w = timeline.width;
  const h = timeline.height;
  const ratio = window.devicePixelRatio || 1;
  tctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const cssW = w / ratio;
  const cssH = h / ratio;
  tctx.clearRect(0, 0, cssW, cssH);
  tctx.fillStyle = "#7dff6a";
  tctx.font = "11px Consolas, monospace";
  tctx.fillText("CHECKPOINTS", 8, 12);
  if (!rewards.length && !checkpoints.length) {
    tctx.fillText("NO CHECKPOINTS YET", 8, cssH - 8);
    return;
  }
  const values = rewards.map((row) => row.reward);
  let min = values.length ? Math.min(...values) : 0;
  let max = values.length ? Math.max(...values) : 1;
  checkpoints.forEach((row) => {
    const value = row.mean_reward ?? row.episode_reward ?? 0;
    min = Math.min(min, value);
    max = Math.max(max, value);
  });
  if (Math.abs(max - min) < 1) {
    min -= 1;
    max += 1;
  }
  const yOf = (v) => (cssH - 14) - ((v - min) / (max - min)) * Math.max(8, cssH - 28);
  if (rewards.length > 1) {
    tctx.beginPath();
    rewards.forEach((row, i) => {
      const x = timelineX(row.episode, cssW);
      const y = yOf(row.reward);
      if (i === 0) tctx.moveTo(x, y);
      else tctx.lineTo(x, y);
    });
    tctx.strokeStyle = "rgba(125, 255, 106, 0.7)";
    tctx.lineWidth = 1;
    tctx.stroke();
  }
  checkpoints.forEach((row) => {
    const x = timelineX(row.episode, cssW);
    const y = yOf(row.mean_reward ?? row.episode_reward ?? 0);
    const active = row.id === selected || row.id === compareA || row.id === compareB;
    tctx.fillStyle = active ? "#f3fff2" : "#7dff6a";
    tctx.fillRect(x - 3, y - 7, 6, 14);
    tctx.fillText(String(row.episode), x - 10, cssH - 2);
  });
}

function checkpointFromEvent(event) {
  const rect = timeline.getBoundingClientRect();
  const x = event.clientX - rect.left;
  let bestHit = null;
  let bestDist = 16;
  checkpoints.forEach((row) => {
    const px = timelineX(row.episode, rect.width);
    const dist = Math.abs(px - x);
    if (dist < bestDist) {
      bestHit = row;
      bestDist = dist;
    }
  });
  return bestHit;
}

function drawHorizon(state) {
  const ratio = window.devicePixelRatio || 1;
  const w = horizon.width;
  const h = horizon.height;
  hctx.setTransform(1, 0, 0, 1, 0, 0);
  hctx.clearRect(0, 0, w, h);
  if (!state) return;
  const cx = w * 0.5;
  const cy = h * 0.40;
  const green = "rgba(125, 255, 106, 0.92)";
  hctx.save();
  hctx.translate(cx, cy);
  hctx.rotate(-state.roll);
  hctx.translate(0, state.pitch * 280 * ratio);
  hctx.strokeStyle = green;
  hctx.fillStyle = green;
  hctx.lineWidth = 1.3 * ratio;
  hctx.beginPath();
  hctx.moveTo(-220 * ratio, 0);
  hctx.lineTo(-34 * ratio, 0);
  hctx.moveTo(34 * ratio, 0);
  hctx.lineTo(220 * ratio, 0);
  hctx.stroke();
  for (const deg of [-30, -20, -10, 10, 20, 30]) {
    const y = ((-deg * Math.PI) / 180) * 280 * ratio;
    const half = Math.abs(deg) >= 20 ? 52 : 30;
    hctx.beginPath();
    hctx.moveTo(-half * ratio, y);
    hctx.lineTo(half * ratio, y);
    hctx.stroke();
    hctx.font = `${11 * ratio}px Consolas, monospace`;
    hctx.fillText(String(Math.abs(deg)), (half + 6) * ratio, y + 4 * ratio);
  }
  hctx.restore();
  hctx.strokeStyle = green;
  hctx.lineWidth = 1.4 * ratio;
  hctx.beginPath();
  hctx.arc(cx, cy, 11 * ratio, 0, Math.PI * 2);
  hctx.moveTo(cx - 28 * ratio, cy);
  hctx.lineTo(cx - 14 * ratio, cy);
  hctx.moveTo(cx + 14 * ratio, cy);
  hctx.lineTo(cx + 28 * ratio, cy);
  hctx.moveTo(cx, cy - 16 * ratio);
  hctx.lineTo(cx, cy - 12 * ratio);
  hctx.stroke();
}

function applyState(state) {
  lastState = state;
  if (state.best) updateBest(state.best);
}

function steerCamera(state, axes) {
  const pos = new THREE.Vector3(state.position[0], state.position[1], state.position[2]);
  let desired;
  if (camMode === "overview") {
    desired = new THREE.Vector3(0, 640, 860);
    lookAt.set(0, 50, 0);
  } else {
    desired = pos.clone().addScaledVector(axes.forward, -20);
    desired.y += 5.2;
    lookAt.copy(pos).addScaledVector(axes.forward, 14);
    lookAt.y += 2.4;
  }
  camera.up.set(0, 1, 0);
  const alpha = cameraSnap > 0 ? 1 : (camMode === "overview" ? 0.16 : 0.14);
  if (cameraSnap > 0) cameraSnap -= 1;
  smoothedCam.lerp(desired, alpha);
  camera.position.copy(smoothedCam);
  camera.lookAt(lookAt);
}

function syncFlyButton() {
  ui.fly.classList.toggle("active", flyIntent);
  ui.fly.setAttribute("aria-pressed", flyIntent ? "true" : "false");
  ui.keys.hidden = !flyIntent;
}

function syncCameraButtons() {
  ui.chase.classList.toggle("active", camMode === "chase");
  ui.overview.classList.toggle("active", camMode === "overview");
  ui.chase.setAttribute("aria-pressed", camMode === "chase" ? "true" : "false");
  ui.overview.setAttribute("aria-pressed", camMode === "overview" ? "true" : "false");
}

function setCamera(mode) {
  camMode = mode;
  cameraSnap = 6;
  syncCameraButtons();
}

function adoptServerMode(mode) {
  if (!mode) return;
  if (lastState) lastState.mode = mode;
  if (flyPending) return;
  const manual = mode === "manual";
  if (manual === flyIntent) return;
  flyIntent = manual;
  syncFlyButton();
}

function upsertReward(row) {
  const index = rewards.findIndex((item) => item.episode === row.episode);
  if (index >= 0) rewards[index] = row;
  else rewards.push(row);
  rewards.sort((a, b) => a.episode - b.episode);
}

function upsertCheckpoint(row) {
  if (!row || !row.id) return;
  const index = checkpoints.findIndex((item) => item.id === row.id);
  if (index >= 0) checkpoints[index] = row;
  else checkpoints.push(row);
  checkpoints.sort((a, b) => a.episode - b.episode);
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || response.statusText;
    ui.compareRead.textContent = String(detail).toUpperCase();
    throw new Error(detail);
  }
  return payload;
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws`);
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      rewards.splice(0, rewards.length, ...(message.rewards || []));
      checkpoints.splice(0, checkpoints.length, ...(message.checkpoints || []));
      updateBest(message.best);
      if (!gateMeshes.length) buildTrack(message.track);
      if (message.status) {
        ui.mode.textContent = (message.status.mode || "idle").toUpperCase();
        adoptServerMode(message.status.mode);
      }
      drawReward();
      drawTimeline();
      if (!lastState && trackStart) {
        applyState({
          position: trackStart.position,
          yaw: trackStart.yaw,
          pitch: trackStart.pitch,
          roll: 0,
          speed: 42,
          altitude: trackStart.position[1],
          g_force: 1,
          throttle: trackStart.throttle,
          gate_index: 0,
          cleared: Array(gateCount).fill(false),
          lap_time: null,
          lap_running: false,
          reason: "",
          mode: "idle",
          playback: 1,
        });
      }
    } else if (message.type === "state") {
      applyState(message);
    } else if (message.type === "metric") {
      upsertReward(message);
      if (message.checkpoint) {
        upsertCheckpoint({
          id: message.checkpoint,
          episode: message.episode,
          mean_reward: message.mean_reward,
          episode_reward: message.reward,
          lap_time: message.lap_time,
        });
      }
      if (message.best) updateBest(message.best);
      drawReward();
      drawTimeline();
    } else if (message.type === "status") {
      ui.mode.textContent = `${message.mode || "idle"}`.toUpperCase();
      ui.stop.hidden = !message.training;
      const progressButton = document.getElementById("progress");
      if (progressButton) {
        const touring = String(message.detail || "").startsWith("Checkpoint tour");
        progressButton.classList.toggle("active", touring);
        progressButton.setAttribute("aria-pressed", touring ? "true" : "false");
      }
      adoptServerMode(message.mode);
      if (message.best) updateBest(message.best);
    } else if (message.type === "compare") {
      if (message.phase === "a") ui.compareRead.textContent = `A ${message.checkpoint}  THEN  B ${message.against}`;
      if (message.phase === "b") ui.compareRead.textContent = `B ${message.checkpoint}`;
      if (message.phase === "done") ui.compareRead.textContent = `COMPARED ${message.a} / ${message.b}`;
    } else if (message.type === "replay_done" && message.best) {
      updateBest(message.best);
    } else if (message.type === "manual_lap" && message.best) {
      updateBest(message.best);
    } else if (message.type === "checkpoints") {
      checkpoints.splice(0, checkpoints.length, ...(message.checkpoints || []));
      drawTimeline();
    } else if (message.type === "training_finished") {
      ui.stop.hidden = true;
      (message.checkpoints || []).forEach(upsertCheckpoint);
      if (message.best) updateBest(message.best);
      drawReward();
      drawTimeline();
    }
  };
  ws.onclose = () => {
    ui.mode.textContent = "LINK LOST";
    setTimeout(connect, 1200);
  };
}

function sendStick() {
  if (!flyIntent || !ws || ws.readyState !== WebSocket.OPEN) return;
  const down = (code) => keys.has(code);
  ws.send(JSON.stringify({
    type: "control",
    throttle: (down("KeyW") ? 1 : 0) - (down("KeyS") ? 1 : 0),
    pitch: (down("ArrowUp") ? 1 : 0) - (down("ArrowDown") ? 1 : 0),
    roll: (down("ArrowRight") ? 1 : 0) - (down("ArrowLeft") ? 1 : 0),
    yaw: (down("KeyD") ? 1 : 0) - (down("KeyA") ? 1 : 0),
  }));
}

document.getElementById("controls").addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  button.blur();
  if (button.dataset.speed) {
    playback = Number(button.dataset.speed);
    document.querySelectorAll("[data-speed]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    await post("/api/speed", { speed: playback });
    return;
  }
  if (button.id === "fly") {
    if (flyPending) return;
    const next = !flyIntent;
    flyIntent = next;
    flyPending = true;
    syncFlyButton();
    try {
      const result = await post("/api/fly", { enabled: next });
      flyIntent = result.mode === "manual";
    } catch (err) {
      flyIntent = !next;
    } finally {
      flyPending = false;
      syncFlyButton();
    }
    return;
  }
  if (button.id === "chase" || button.id === "overview") {
    setCamera(button.id);
    return;
  }
  if (button.id === "progress") {
    button.classList.add("active");
    button.setAttribute("aria-pressed", "true");
    await post("/api/progress", { speed: playback });
    return;
  }
  if (button.id === "compare") {
    compareArmed = !compareArmed;
    compareA = null;
    compareB = null;
    ui.compare.classList.toggle("active", compareArmed);
    ui.compare.setAttribute("aria-pressed", compareArmed ? "true" : "false");
    ui.compareRead.textContent = compareArmed ? "SELECT CHECKPOINT A" : "";
    drawTimeline();
    return;
  }
  if (button.id === "train") {
    ui.confirm.hidden = false;
    await refreshPlan();
    return;
  }
  if (button.id === "stop") {
    await post("/api/train/stop", {});
  }
});

timeline.addEventListener("click", async (event) => {
  const row = checkpointFromEvent(event);
  if (!row) return;
  selected = row.id;
  drawTimeline();
  if (compareArmed) {
    if (!compareA) {
      compareA = row.id;
      ui.compareRead.textContent = `A ${row.id}   SELECT B`;
      return;
    }
    compareB = row.id;
    compareArmed = false;
    ui.compare.classList.remove("active");
    ui.compare.setAttribute("aria-pressed", "false");
    ui.compareRead.textContent = `A ${compareA}  THEN  B ${compareB}`;
    await post("/api/compare", { a: compareA, b: compareB, speed: playback });
    return;
  }
  await post("/api/replay", { checkpoint: row.id, speed: playback });
});

document.getElementById("episodes").addEventListener("change", refreshPlan);
document.getElementById("every").addEventListener("change", refreshPlan);
document.getElementById("confirmCancel").addEventListener("click", () => {
  ui.confirm.hidden = true;
});
document.getElementById("confirmStart").addEventListener("click", async () => {
  const episodes = Number(document.getElementById("episodes").value);
  const checkpoint_every = Number(document.getElementById("every").value);
  ui.confirm.hidden = true;
  await post("/api/train", { confirm: true, episodes, checkpoint_every, envs: 4 });
  ui.stop.hidden = false;
});

async function refreshPlan() {
  const episodes = Number(document.getElementById("episodes").value) || 200;
  const every = Number(document.getElementById("every").value) || 20;
  const response = await fetch(`/api/plan?episodes=${episodes}&checkpoint_every=${every}&envs=4`);
  const payload = await response.json();
  ui.plan.textContent = payload.plan;
}

window.addEventListener("keydown", (event) => {
  if (event.target && (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA")) return;
  keys.add(event.code);
  if (flyIntent && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyW", "KeyS", "KeyA", "KeyD"].includes(event.code)) {
    event.preventDefault();
  }
});
window.addEventListener("keyup", (event) => {
  keys.delete(event.code);
});
window.addEventListener("blur", () => keys.clear());
window.addEventListener("resize", resize);

function frame() {
  requestAnimationFrame(frame);
  if (lastState) {
    const axes = updateHud(lastState);
    steerCamera(lastState, axes);
    drawHorizon(lastState);
  }
  renderer.render(scene, camera);
}

syncCameraButtons();
document.querySelector("[data-speed='1']").classList.add("active");
resize();
connect();
setInterval(sendStick, 50);
frame();
