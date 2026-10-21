import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { bodyAxes, type Track, type Vec3 } from "../sim/jet";

/**
 * Daylight airfield: sky, clouds, snowy mountains outside the airspace, a
 * runway under the start line, and racing gates on lattice towers.
 * Sim coordinates are mirrored relative to three.js handedness; everything
 * the sim knows about lives under `root`, which flips x back.
 */

export const HUD_GREEN = "#7dff9a";

export interface Pose {
  position: Vec3;
  yaw: number;
  pitch: number;
  roll: number;
}

export type CameraMode = "chase" | "cockpit" | "tv" | "orbit" | "overview";

const FLAT_RADIUS = 1600; // flat airfield inside this; the airspace never reaches the mountains


function hash(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function valueNoise(x: number, y: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi);
  const b = hash(xi + 1, yi);
  const c = hash(xi, yi + 1);
  const d = hash(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x: number, y: number, oct = 5): number {
  let amp = 0.5;
  let f = 1;
  let sum = 0;
  for (let i = 0; i < oct; i += 1) {
    sum += amp * valueNoise(x * f, y * f);
    f *= 2.03;
    amp *= 0.5;
  }
  return sum;
}


function canvasTexture(w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void, repeat = false): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d")!);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

const skyTexture = () =>
  canvasTexture(2, 512, (ctx) => {
    const g = ctx.createLinearGradient(0, 0, 0, 512);
    g.addColorStop(0, "#2f6fb8");
    g.addColorStop(0.42, "#78acdc");
    g.addColorStop(0.5, "#c9def0");
    g.addColorStop(0.52, "#dfe8ee");
    g.addColorStop(1, "#c9d5de");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 2, 512);
  });

const cloudTexture = () =>
  canvasTexture(256, 128, (ctx) => {
    for (let i = 0; i < 46; i += 1) {
      const x = 40 + Math.random() * 176;
      const y = 50 + Math.random() * 40 - Math.abs(x - 128) * 0.15;
      const r = 14 + Math.random() * 30;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      const shade = 235 + Math.random() * 20;
      g.addColorStop(0, `rgba(${shade},${shade},${shade + 2},0.55)`);
      g.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  });

const snowTexture = () =>
  canvasTexture(
    512,
    512,
    (ctx) => {
      const img = ctx.createImageData(512, 512);
      for (let y = 0; y < 512; y += 1) {
        for (let x = 0; x < 512; x += 1) {
          const n = fbm(x / 64, y / 64, 4);
          const m = valueNoise(x / 9, y / 9);
          const v = 200 + n * 40 + m * 12;
          const i = (y * 512 + x) * 4;
          img.data[i] = v - 6;
          img.data[i + 1] = v - 2;
          img.data[i + 2] = v + 4;
          img.data[i + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    },
    true,
  );

const runwayTexture = () =>
  canvasTexture(256, 2048, (ctx) => {
    ctx.fillStyle = "#5c6066";
    ctx.fillRect(0, 0, 256, 2048);
    for (let i = 0; i < 3000; i += 1) {
      ctx.fillStyle = `rgba(${Math.random() < 0.5 ? 0 : 255},${Math.random() < 0.5 ? 0 : 255},255,0.03)`;
      ctx.fillRect(Math.random() * 256, Math.random() * 2048, 2, 2);
    }
    ctx.fillStyle = "#e8eaec";
    for (let y = 120; y < 1900; y += 90) ctx.fillRect(124, y, 8, 50); // centerline
    ctx.fillRect(10, 0, 4, 2048); // edges
    ctx.fillRect(242, 0, 4, 2048);
    for (let k = 0; k < 8; k += 1) ctx.fillRect(28 + k * 26, 2000, 14, 40); // threshold
    for (let k = 0; k < 8; k += 1) ctx.fillRect(28 + k * 26, 8, 14, 40);
  });

const stripeTexture = () =>
  canvasTexture(
    512,
    8,
    (ctx) => {
      for (let i = 0; i < 16; i += 1) {
        ctx.fillStyle = i % 2 ? "#f2f2ef" : "#d8452a";
        ctx.fillRect(i * 32, 0, 32, 8);
      }
    },
    true,
  );


function buildJetMesh(accent: number, opacity: number): { group: THREE.Group; burner: THREE.Mesh } {
  const g = new THREE.Group();
  const transparent = opacity < 1;
  const skin = new THREE.MeshStandardMaterial({ color: 0x7c848d, metalness: 0.55, roughness: 0.38, transparent, opacity });
  const dark = new THREE.MeshStandardMaterial({ color: 0x3a4048, metalness: 0.4, roughness: 0.5, transparent, opacity });
  const glass = new THREE.MeshStandardMaterial({ color: 0x2a2416, metalness: 0.9, roughness: 0.08, transparent: true, opacity: Math.min(opacity, 0.9) });
  const mark = new THREE.MeshBasicMaterial({ color: accent, transparent, opacity });

  // Fuselage: lathe profile along z, nose at +z.
  const prof: THREE.Vector2[] = [];
  const L = 15;
  for (let i = 0; i <= 24; i += 1) {
    const t = i / 24;
    const r = t < 0.28 ? 0.05 + 0.95 * Math.sin((t / 0.28) * Math.PI * 0.5) ** 0.8 : t < 0.85 ? 1 : 1 - (t - 0.85) * 1.6;
    prof.push(new THREE.Vector2(Math.max(0.02, r * 0.95), (0.5 - t) * L));
  }
  const fus = new THREE.Mesh(new THREE.LatheGeometry(prof, 20), skin);
  fus.rotation.x = Math.PI / 2;
  fus.scale.set(1.15, 0.82, 1);
  g.add(fus);
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.7, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2), glass);
  canopy.scale.set(0.9, 0.85, 2.6);
  canopy.position.set(0, 0.62, 3.2);
  g.add(canopy);
  for (const s of [-1, 1]) {
    const intake = new THREE.Mesh(new THREE.BoxGeometry(0.75, 1.0, 4.2), dark);
    intake.position.set(s * 1.3, -0.05, 1.0);
    intake.rotation.y = s * -0.06;
    g.add(intake);
  }
  // Wings: trapezoid, swept leading edge.
  const wing = new THREE.Shape();
  wing.moveTo(0, 2.6);
  wing.lineTo(5.6, -1.4);
  wing.lineTo(5.6, -2.6);
  wing.lineTo(0, -3.2);
  wing.closePath();
  const wingGeo = new THREE.ExtrudeGeometry(wing, { depth: 0.16, bevelEnabled: false });
  wingGeo.rotateX(Math.PI / 2);
  for (const s of [-1, 1]) {
    const w = new THREE.Mesh(wingGeo, skin);
    w.scale.x = s;
    w.position.set(s * 0.6, -0.1, 0);
    g.add(w);
    const tip = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 1.0), mark);
    tip.position.set(s * 6.15, -0.1, -1.9);
    g.add(tip);
  }
  // Horizontal stabilators.
  const stab = new THREE.Shape();
  stab.moveTo(0, 0.6);
  stab.lineTo(2.8, -0.9);
  stab.lineTo(2.8, -1.6);
  stab.lineTo(0, -1.8);
  stab.closePath();
  const stabGeo = new THREE.ExtrudeGeometry(stab, { depth: 0.12, bevelEnabled: false });
  stabGeo.rotateX(Math.PI / 2);
  for (const s of [-1, 1]) {
    const h = new THREE.Mesh(stabGeo, skin);
    h.scale.x = s;
    h.position.set(s * 0.9, 0, -5.4);
    g.add(h);
  }
  // Twin canted vertical tails.
  const fin = new THREE.Shape();
  fin.moveTo(0, 0);
  fin.lineTo(2.6, 0);
  fin.lineTo(1.0, 2.9);
  fin.lineTo(-0.1, 2.9);
  fin.closePath();
  const finGeo = new THREE.ExtrudeGeometry(fin, { depth: 0.12, bevelEnabled: false });
  finGeo.rotateY(Math.PI / 2);
  for (const s of [-1, 1]) {
    const f = new THREE.Mesh(finGeo, skin);
    f.position.set(s * 1.1, 0.5, -3.6);
    f.rotation.z = s * -0.38;
    g.add(f);
    const band = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.35, 1.2), mark);
    band.position.set(s * 1.75, 2.4, -4.6);
    band.rotation.z = s * -0.38;
    g.add(band);
  }
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.72, 1.2, 16, 1, true), dark);
  nozzle.rotation.x = Math.PI / 2;
  nozzle.position.z = -7.7;
  g.add(nozzle);
  const burner = new THREE.Mesh(
    new THREE.ConeGeometry(0.5, 2.4, 16, 1, true),
    new THREE.MeshBasicMaterial({ color: 0xff7a2e, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }),
  );
  burner.rotation.x = -Math.PI / 2;
  burner.position.z = -10;
  g.add(burner);
  g.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && o !== burner) (o as THREE.Mesh).castShadow = true;
  });
  return { group: g, burner };
}

/** Fading ribbon of past positions. The age ramp is precomputed. */
class Trail {
  readonly line: THREE.Line;
  private readonly positions: Float32Array;
  private readonly ramp: Float32Array;
  private count = 0;

  constructor(color: number, private readonly size = 300) {
    this.positions = new Float32Array(size * 3);
    this.ramp = new Float32Array(size * 4);
    const base = new THREE.Color(color).lerp(new THREE.Color(0xffffff), 0.45);
    for (let i = 0; i < size; i += 1) this.ramp.set([base.r, base.g, base.b, (i / size) ** 1.6 * 0.9], i * 4);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(size * 4), 4));
    geo.setDrawRange(0, 0);
    this.line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    this.line.frustumCulled = false;
  }

  clear(): void {
    this.count = 0;
    this.line.geometry.setDrawRange(0, 0);
  }

  push(p: Vec3): void {
    if (this.count === this.size) {
      this.positions.copyWithin(0, 3);
      this.count -= 1;
    }
    this.positions.set(p, this.count * 3);
    this.count += 1;
    const geo = this.line.geometry;
    geo.setDrawRange(0, this.count);
    const colors = geo.attributes.color as THREE.BufferAttribute;
    (colors.array as Float32Array).set(this.ramp.subarray((this.size - this.count) * 4), 0);
    geo.attributes.position.needsUpdate = true;
    colors.needsUpdate = true;
  }
}

export class JetVisual {
  readonly group: THREE.Group;
  readonly trail: Trail;
  private readonly burner: THREE.Mesh;
  private readonly prev = new THREE.Vector3();
  private readonly cur = new THREE.Vector3();
  private readonly qPrev = new THREE.Quaternion();
  private readonly qCur = new THREE.Quaternion();
  /** Sim-space travel direction, from the last two committed steps. */
  readonly heading = new THREE.Vector3(0, 0, 1);
  private throttle = 0.5;
  private primed = false;

  constructor(root: THREE.Object3D, readonly color: number, opacity = 1) {
    const built = buildJetMesh(color, opacity);
    this.group = built.group;
    this.burner = built.burner;
    this.trail = new Trail(color);
    const dot = new THREE.Points(
      new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0], 3)),
      new THREE.PointsMaterial({ color, size: 9, sizeAttenuation: false, depthTest: false, transparent: true }),
    );
    dot.renderOrder = 10;
    this.group.add(dot);
    root.add(this.group, this.trail.line);
  }

  commit(p: Pose & { throttle?: number }): void {
    this.prev.copy(this.cur);
    this.qPrev.copy(this.qCur);
    this.cur.set(...p.position);
    const { forward, right, up } = bodyAxes(p.yaw, p.pitch, p.roll);
    this.qCur.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3(...right), new THREE.Vector3(...up), new THREE.Vector3(...forward)));
    if (!this.primed) {
      this.prev.copy(this.cur);
      this.qPrev.copy(this.qCur);
      this.primed = true;
    }
    const d = this.cur.clone().sub(this.prev);
    if (d.lengthSq() > 1e-6) this.heading.copy(d.normalize());
    if (p.throttle !== undefined) this.throttle = p.throttle;
    this.trail.push(p.position);
  }

  draw(alpha: number, time: number): void {
    this.group.position.lerpVectors(this.prev, this.cur, alpha);
    this.group.quaternion.slerpQuaternions(this.qPrev, this.qCur, alpha);
    const flicker = 1 + Math.sin(time * 47) * 0.06;
    const len = 0.3 + this.throttle * 0.9 * flicker;
    this.burner.scale.set(1, len, 1);
    this.burner.position.z = -8.3 - len * 1.2;
  }

  reset(): void {
    this.primed = false;
    this.trail.clear();
    this.group.visible = true;
  }

  dispose(): void {
    this.group.removeFromParent();
    this.trail.line.removeFromParent();
  }
}


function labelSprite(text: string): THREE.Sprite {
  const tex = canvasTexture(128, 64, (ctx) => {
    ctx.font = "700 44px 'IBM Plex Sans Condensed', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(20,26,34,0.85)";
    ctx.strokeText(text, 64, 34);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, 64, 34);
  });
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  s.scale.set(16, 8, 1);
  return s;
}

/** Lattice tower with silos on top, like an industrial pylon. One merged mesh. */
function towerGeometry(height: number): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const w = 7;
  for (const [x, z] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    const leg = new THREE.BoxGeometry(0.6, height, 0.6);
    leg.translate((x * w) / 2, height / 2, (z * w) / 2);
    parts.push(leg);
  }
  for (let y = 4; y < height; y += 6) {
    for (const [ax, az, rot] of [[0, -w / 2, 0], [0, w / 2, 0], [-w / 2, 0, Math.PI / 2], [w / 2, 0, Math.PI / 2]] as const) {
      const bar = new THREE.BoxGeometry(w, 0.35, 0.35);
      bar.rotateY(rot);
      bar.translate(ax, y, az);
      parts.push(bar);
      const brace = new THREE.BoxGeometry(Math.hypot(w, 6), 0.25, 0.25);
      brace.rotateZ(Math.atan2(6, w) * (Math.round(y / 6) % 2 ? 1 : -1));
      brace.rotateY(rot);
      brace.translate(ax, y + 3, az);
      if (y + 6 < height) parts.push(brace);
    }
  }
  const deck = new THREE.BoxGeometry(w + 2, 0.8, w + 2);
  deck.translate(0, height, 0);
  parts.push(deck);
  for (const [x, z] of [[-1.8, 0], [1.8, 0], [0, 1.9]]) {
    const silo = new THREE.CylinderGeometry(1.6, 1.6, 9, 14);
    silo.translate(x, height + 5, z);
    parts.push(silo);
  }
  return mergeGeometries(parts)!;
}

interface GateVisual {
  ring: THREE.MeshStandardMaterial;
  halo: THREE.Mesh;
}


export class World {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(62, 1, 0.5, 9000);
  readonly root = new THREE.Group();
  private trackGroup = new THREE.Group();
  private gates: GateVisual[] = [];
  private readonly controls: OrbitControls;
  private readonly camPos = new THREE.Vector3(0, 300, -600);
  private readonly camLook = new THREE.Vector3();
  private readonly camDir = new THREE.Vector3(0, 0, 1);
  private readonly sun: THREE.DirectionalLight;
  private readonly targetLine: THREE.Line;
  private readonly stripes = (() => {
    const t = stripeTexture();
    t.repeat.set(3, 1);
    return t;
  })();
  private readonly towerMat = new THREE.MeshStandardMaterial({ color: 0x8a7a5c, metalness: 0.5, roughness: 0.6 });
  private readonly siloMat = new THREE.MeshStandardMaterial({ color: 0x4a4f57, metalness: 0.6, roughness: 0.45 });
  private time = 0;
  mode: CameraMode = "chase";
  target: JetVisual | null = null;
  track: Track;

  constructor(private readonly canvas: HTMLCanvasElement, track: Track) {
    this.track = track;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.background = skyTexture();
    this.scene.fog = new THREE.Fog(0xcfdde9, 1400, 6500);
    this.root.scale.x = -1;
    this.scene.add(this.root);

    this.scene.add(new THREE.HemisphereLight(0xdbeaff, 0x8e8a80, 1.15));
    this.sun = new THREE.DirectionalLight(0xfff4e0, 2.4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -60, right: 60, top: 60, bottom: -60, near: 10, far: 900 });
    this.sun.shadow.bias = -0.0004;
    this.scene.add(this.sun, this.sun.target);
    const sunDisc = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: canvasTexture(128, 128, (ctx) => {
          const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
          g.addColorStop(0, "rgba(255,255,245,1)");
          g.addColorStop(0.18, "rgba(255,250,230,0.9)");
          g.addColorStop(1, "rgba(255,240,210,0)");
          ctx.fillStyle = g;
          ctx.fillRect(0, 0, 128, 128);
        }),
        fog: false,
        transparent: true,
        depthWrite: false,
      }),
    );
    sunDisc.scale.set(900, 900, 1);
    sunDisc.position.set(-3500, 2600, 3800);
    this.scene.add(sunDisc);

    this.buildTerrain();
    this.buildClouds();

    const lineGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 1], 3));
    this.targetLine = new THREE.Line(lineGeo, new THREE.LineDashedMaterial({ color: HUD_GREEN, dashSize: 5, gapSize: 4, transparent: true, opacity: 0.9 }));
    this.targetLine.frustumCulled = false;
    this.root.add(this.targetLine);

    this.setTrack(track);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.maxDistance = 3000;
    this.controls.enabled = false;
    this.resize();
  }

  private buildTerrain(): void {
    const size = 9000;
    const seg = 220;
    const geo = new THREE.PlaneGeometry(size, size, seg, seg);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(pos.count * 3);
    const rock = new THREE.Color(0x6d6a66);
    const snow = new THREE.Color(0xf4f7fa);
    const field = new THREE.Color(0xe6ebef);
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i += 1) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const r = Math.hypot(x, z);
      const rise = Math.max(0, (r - FLAT_RADIUS) / 1300);
      const ridge = 1 - Math.abs(fbm(x / 900 + 3, z / 900 - 7, 5) * 2 - 1);
      const h = r < FLAT_RADIUS ? 0 : Math.min(1, rise) ** 1.6 * (110 + ridge * ridge * 360) + fbm(x / 120, z / 120, 3) * 40 * Math.min(1, rise);
      pos.setY(i, h - 0.5);
      const snowLine = 90 + fbm(x / 300, z / 300, 3) * 90;
      if (h < 2) c.copy(field);
      else c.copy(rock).lerp(snow, Math.min(1, Math.max(0, (h - snowLine) / 90)));
      colors.set([c.r, c.g, c.b], i * 3);
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const tex = snowTexture();
    tex.repeat.set(60, 60);
    const ground = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, map: tex }));
    ground.receiveShadow = true;
    this.root.add(ground);
    this.buildTrees();
  }

  /** Sparse pines between the airspace and the mountains: scale and a sense of speed. */
  private buildTrees(): void {
    const count = 1800;
    const geo = new THREE.ConeGeometry(4, 16, 6);
    geo.translate(0, 8, 0);
    const trunk = new THREE.CylinderGeometry(0.6, 0.8, 3, 5);
    trunk.translate(0, -1.4, 0);
    const merged = mergeGeometries([geo, trunk])!;
    const mesh = new THREE.InstancedMesh(merged, new THREE.MeshLambertMaterial({ color: 0x2f4a3a, flatShading: true }), count);
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    let n = 0;
    for (let i = 0; n < count && i < count * 4; i += 1) {
      const a = hash(i, 1.7) * Math.PI * 2;
      const r = 520 + hash(i, 9.1) ** 0.7 * 1050;
      // Clumps: keep trees where the noise says forest.
      const x = Math.cos(a) * r;
      const z = Math.sin(a) * r;
      if (fbm(x / 220, z / 220, 3) < 0.5) continue;
      const sc = 0.7 + hash(i, 4.4) * 0.9;
      m.compose(new THREE.Vector3(x, 0, z), q, new THREE.Vector3(sc, sc * (0.9 + hash(i, 2.2) * 0.5), sc));
      mesh.setMatrixAt(n, m);
      n += 1;
    }
    mesh.count = n;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    this.root.add(mesh);
  }

  private buildClouds(): void {
    const mats = [0, 1, 2].map(() => new THREE.SpriteMaterial({ map: cloudTexture(), transparent: true, depthWrite: false, opacity: 0.9 }));
    for (let i = 0; i < 70; i += 1) {
      const a = Math.random() * Math.PI * 2;
      const r = 600 + Math.random() * 3600;
      const s = new THREE.Sprite(mats[i % 3]);
      const w = 300 + Math.random() * 600;
      s.scale.set(w, w * 0.45, 1);
      s.position.set(Math.cos(a) * r, 380 + Math.random() * 520, Math.sin(a) * r);
      this.root.add(s);
    }
  }

  setTrack(track: Track): void {
    this.track = track;
    this.root.remove(this.trackGroup);
    this.trackGroup.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
    });
    this.trackGroup = new THREE.Group();
    this.gates = [];
    const g = this.trackGroup;

    // Runway under the start line, pointing along the start heading.
    const st = track.start;
    const runway = new THREE.Mesh(new THREE.PlaneGeometry(46, 1400), new THREE.MeshStandardMaterial({ map: runwayTexture(), roughness: 0.9 }));
    runway.rotation.x = -Math.PI / 2;
    // Long axis (+y before laying flat) ends up along the start heading in sim space.
    runway.rotation.z = st.yaw;
    const back = 560;
    runway.position.set(st.position[0] - Math.sin(st.yaw) * back, 0.2, st.position[2] - Math.cos(st.yaw) * back);
    runway.receiveShadow = true;
    g.add(runway);

    // Faint racing line in the air.
    const air = track.centerline.flatMap((p) => p);
    air.push(...track.centerline[0]);
    const airLine = new THREE.Line(
      new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(air, 3)),
      new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 8, gapSize: 10, transparent: true, opacity: 0.35 }),
    );
    airLine.computeLineDistances();
    g.add(airLine);

    track.centers.forEach((c, i) => {
      const n = track.normals[i];
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...n).normalize());
      const ringMat = new THREE.MeshStandardMaterial({ map: this.stripes, color: 0xffffff, roughness: 0.55, metalness: 0.1, emissive: 0x000000 });
      const ring = new THREE.Mesh(new THREE.TorusGeometry(track.radius, 1.1, 12, 96), ringMat);
      ring.position.set(...c);
      ring.quaternion.copy(q);
      ring.castShadow = true;
      g.add(ring);
      const halo = new THREE.Mesh(
        new THREE.TorusGeometry(track.radius + 3.5, 0.45, 8, 96),
        new THREE.MeshBasicMaterial({ color: HUD_GREEN, transparent: true, opacity: 0.55, fog: false }),
      );
      halo.position.set(...c);
      halo.quaternion.copy(q);
      halo.visible = false;
      g.add(halo);
      // Towers either side of the hoop, beams up to it.
      const side = new THREE.Vector3(n[2], 0, -n[0]).normalize();
      for (const s of [-1, 1]) {
        const base = new THREE.Vector3(...c).addScaledVector(side, s * (track.radius + 7));
        const towerH = Math.max(8, c[1] - 6);
        const tower = new THREE.Mesh(towerGeometry(towerH), this.towerMat);
        tower.position.set(base.x, 0, base.z);
        tower.rotation.y = Math.atan2(n[0], n[2]);
        tower.castShadow = true;
        tower.receiveShadow = true;
        g.add(tower);
        const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 7, 8), this.siloMat);
        beam.position.copy(new THREE.Vector3(...c).addScaledVector(side, s * (track.radius + 3.5)));
        beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), side);
        g.add(beam);
      }
      const label = labelSprite(String(i + 1));
      label.position.set(c[0], c[1] + track.radius + 9, c[2]);
      g.add(label);
      this.gates.push({ ring: ringMat, halo });
    });
    this.root.add(g);
  }

  /** Highlight the next gate for the followed jet and draw its target line. */
  setGates(next: number, cleared: readonly boolean[], from: Vec3 | null): void {
    this.gates.forEach((gv, i) => {
      gv.halo.visible = i === next;
      gv.ring.emissive.setHex(cleared[i] && i !== next ? 0x0b2214 : 0x000000);
      if (gv.halo.visible) gv.halo.scale.setScalar(1 + 0.05 * Math.sin(this.time * 5));
    });
    this.targetLine.visible = !!from && next < this.track.centers.length;
    if (from && this.targetLine.visible) {
      const c = this.track.centers[next];
      const pos = this.targetLine.geometry.attributes.position as THREE.BufferAttribute;
      pos.setXYZ(0, ...from);
      pos.setXYZ(1, ...c);
      pos.needsUpdate = true;
      this.targetLine.computeLineDistances();
    }
  }

  setMode(mode: CameraMode): void {
    this.mode = mode;
    this.controls.enabled = mode === "orbit";
    if (mode === "orbit" && this.target) this.controls.target.copy(this.target.group.getWorldPosition(new THREE.Vector3()));
  }

  resize(): void {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.fov = this.camera.aspect < 1 ? 78 : 62;
    this.camera.updateProjectionMatrix();
  }

  /** Project a three.js world point to canvas pixels, or null when behind the camera. */
  project(p: THREE.Vector3): { x: number; y: number } | null {
    const v = p.clone().project(this.camera);
    if (v.z > 1) return null;
    return { x: ((v.x + 1) / 2) * this.canvas.clientWidth, y: ((1 - v.y) / 2) * this.canvas.clientHeight };
  }

  private toWorld(p: Vec3): THREE.Vector3 {
    return new THREE.Vector3(-p[0], p[1], p[2]);
  }

  render(dt: number, nextGate: number | null): void {
    this.time += dt;
    const target = this.target;
    if (target) {
      const pos = target.group.getWorldPosition(new THREE.Vector3());
      const vel = new THREE.Vector3(-target.heading.x, target.heading.y, target.heading.z);
      this.camDir.lerp(vel, 1 - Math.exp(-dt * 2.8)).normalize();
      const k = 1 - Math.exp(-dt * 7);
      if (this.mode === "chase") {
        const flat = new THREE.Vector3(this.camDir.x, Math.max(-0.35, Math.min(0.35, this.camDir.y)), this.camDir.z).normalize();
        const desired = pos.clone().addScaledVector(flat, -34).add(new THREE.Vector3(0, 7.5, 0));
        this.camPos.lerp(desired, k);
        this.camLook.lerp(pos.clone().addScaledVector(flat, 40).add(new THREE.Vector3(0, 2, 0)), k);
        this.camera.up.set(0, 1, 0);
        this.camera.position.copy(this.camPos);
        this.camera.lookAt(this.camLook);
      } else if (this.mode === "cockpit") {
        const q = target.group.getWorldQuaternion(new THREE.Quaternion());
        const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
        const up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
        this.camera.position.copy(pos).addScaledVector(fwd, 6.5).addScaledVector(up, 1.9);
        this.camera.up.copy(up);
        this.camera.lookAt(pos.clone().addScaledVector(fwd, 150));
      } else if (this.mode === "tv") {
        const n = nextGate !== null ? this.track.normals[nextGate] : undefined;
        const c = nextGate !== null ? this.track.centers[nextGate] : undefined;
        const g = c ? this.toWorld(c) : pos;
        const side = n ? new THREE.Vector3(n[2], 0, n[0]).normalize().multiplyScalar(55) : new THREE.Vector3(55, 0, 0);
        this.camPos.lerp(g.clone().add(new THREE.Vector3(side.x, 14, side.z)), 1 - Math.exp(-dt * 1.5));
        this.camLook.lerp(pos, k);
        this.camera.up.set(0, 1, 0);
        this.camera.position.copy(this.camPos);
        this.camera.lookAt(this.camLook);
      } else if (this.mode === "overview") {
        let minX = Infinity;
        let maxX = -Infinity;
        let minZ = Infinity;
        let maxZ = -Infinity;
        for (const p of this.track.centerline) {
          minX = Math.min(minX, -p[0]);
          maxX = Math.max(maxX, -p[0]);
          minZ = Math.min(minZ, p[2]);
          maxZ = Math.max(maxZ, p[2]);
        }
        const cx = (minX + maxX) / 2;
        const cz = (minZ + maxZ) / 2;
        const span = Math.max(maxX - minX, maxZ - minZ);
        const e = 1 - Math.exp(-dt * 2);
        this.camPos.lerp(new THREE.Vector3(cx, span * 0.8, cz - span * 0.75), e);
        this.camLook.lerp(new THREE.Vector3(cx, 30, cz), e);
        this.camera.up.set(0, 1, 0);
        this.camera.position.copy(this.camPos);
        this.camera.lookAt(this.camLook);
      } else {
        this.controls.target.lerp(pos, k);
        this.controls.update();
      }
      // Keep the shadow camera on the jet.
      this.sun.position.copy(pos).add(new THREE.Vector3(-300, 500, 300));
      this.sun.target.position.copy(pos);
    }
    this.renderer.render(this.scene, this.camera);
  }
}
