"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Grid, Prediction } from "../lib/types";
import { gridExtent } from "../lib/grid";
import { thermalColor } from "../lib/color";
import { DEPTH_MAP_MAX, LEVEL_SPAN, depthT, levelT, sampleProfile, tToDepth } from "../lib/depth";

// Basin scene: X spans longitude, Z spans latitude, Y is the water column.
const BOX_X = 104;
const BOX_Z = 44;
const BOX_Y = 54;

// Column scene: a single block of water at the probe, 0 m to 2000 m.
const BLK_X = 46;
const BLK_Z = 46;
const BLK_Y = 98;

const PARTICLES = 1100;

export type VolumeMode = "column" | "basin";

export interface VolumeControls {
  /** `column` shows the volumetric water block, `basin` the spatial stack. */
  mode: VolumeMode;
  /** Nearest resolved level to the selected depth. */
  depthIdx: number;
  /** Continuously selected depth in metres, driven by the depth map. */
  depthM: number;
  /** Show the faint horizontal boundary rings between depth strata. */
  showStrata: boolean;
  /** Ambient depth particles drifting inside the block. */
  showParticles: boolean;
  /** Show the horizontal stack of depth-level slices (basin mode). */
  showSlices: boolean;
  /** Show the two vertical cross-section curtains through the picked point. */
  showCurtains: boolean;
  /** Show the active depth indicator plane. */
  showCursor: boolean;
  /** Auto-rotate the camera. */
  spin: boolean;
  /** How many of the levels to draw in the stack (perf / clarity). */
  sliceStride: number;
  /** Overall wall / slice opacity. */
  opacity: number;
}

interface Props {
  pred: Prediction;
  depths: number[];
  controls: VolumeControls;
  picked: { row: number; col: number } | null;
  /** Column profile for the frame currently on screen. */
  column: { predicted: (number | null)[]; truth: (number | null)[] } | null;
  /** Temperature scale shared across the whole time series. */
  scale: { min: number; max: number };
  thermocline: number | null;
}

function yFromT(t: number, height: number): number {
  return height * 0.5 - t * height;
}

type Disposable = { dispose: () => void };

function purge(list: { current: Disposable[] }): void {
  list.current.forEach((d) => d.dispose());
  list.current = [];
}

/** Release every geometry and material hanging off a subtree. */
function disposeTree(root: THREE.Object3D): void {
  root.traverse((o) => {
    const m = o as Partial<THREE.Mesh>;
    m.geometry?.dispose();
    const mat = m.material;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
    else mat?.dispose();
  });
}

/** One horizontal depth level -> RGBA canvas (north at canvas top). */
function levelCanvas(grid: Grid, min: number, max: number, alpha: number): HTMLCanvasElement {
  const H = grid.length;
  const W = grid[0]?.length ?? 0;
  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = H;
  const ctx = cv.getContext("2d");
  if (!ctx || !W) return cv;
  const img = ctx.createImageData(W, H);
  for (let r = 0; r < H; r++) {
    const y = H - 1 - r; // lat ascends with r; canvas row 0 is north
    for (let c = 0; c < W; c++) {
      const v = grid[r][c];
      const i = (y * W + c) * 4;
      if (v === null || Number.isNaN(v)) continue; // land stays transparent
      const [rr, gg, bb] = thermalColor(v, min, max);
      img.data[i] = rr;
      img.data[i + 1] = gg;
      img.data[i + 2] = bb;
      img.data[i + 3] = alpha;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

/** Vertical curtain along a latitude row: x = longitude, y = depth. */
function latCurtainCanvas(volume: Grid[], row: number, min: number, max: number): HTMLCanvasElement {
  const nD = volume.length;
  const W = volume[0][row]?.length ?? 0;
  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = nD;
  const ctx = cv.getContext("2d");
  if (!ctx || !W) return cv;
  const img = ctx.createImageData(W, nD);
  for (let d = 0; d < nD; d++) {
    for (let c = 0; c < W; c++) {
      const v = volume[d][row]?.[c];
      const i = (d * W + c) * 4; // canvas row 0 = shallowest
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      const [rr, gg, bb] = thermalColor(v, min, max);
      img.data[i] = rr;
      img.data[i + 1] = gg;
      img.data[i + 2] = bb;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

/** Vertical curtain along a longitude column: x = latitude (north first), y = depth. */
function lonCurtainCanvas(volume: Grid[], col: number, min: number, max: number): HTMLCanvasElement {
  const nD = volume.length;
  const nLat = volume[0].length;
  const cv = document.createElement("canvas");
  cv.width = nLat;
  cv.height = nD;
  const ctx = cv.getContext("2d");
  if (!ctx) return cv;
  const img = ctx.createImageData(nLat, nD);
  for (let d = 0; d < nD; d++) {
    for (let r = 0; r < nLat; r++) {
      // Local +X of the rotated plane points north, so column 0 is max lat.
      const v = volume[d][nLat - 1 - r]?.[col];
      const i = (d * nLat + r) * 4;
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      const [rr, gg, bb] = thermalColor(v, min, max);
      img.data[i] = rr;
      img.data[i + 1] = gg;
      img.data[i + 2] = bb;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

function texFrom(canvas: HTMLCanvasElement): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  return t;
}

const GRAD_H = 512;

/** Repaint the block's wall gradient in place, so playback never rebuilds GPU state. */
function paintGradient(
  cv: HTMLCanvasElement,
  profile: (number | null)[],
  depths: number[],
  min: number,
  max: number
): void {
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  const W = cv.width;
  ctx.clearRect(0, 0, W, GRAD_H);

  for (let y = 0; y < GRAD_H; y++) {
    const t = y / (GRAD_H - 1);
    const depth = tToDepth(t, depths);
    const v = sampleProfile(profile, depths, depth);
    if (t <= LEVEL_SPAN && v !== null) {
      const [r, g, b] = thermalColor(v, min, max);
      // Deepen slightly with depth so the block reads as a solid body of water.
      const k = 1 - (t / LEVEL_SPAN) * 0.18;
      ctx.fillStyle = `rgb(${Math.round(r * k)}, ${Math.round(g * k)}, ${Math.round(b * k)})`;
      ctx.globalAlpha = 1;
    } else {
      // Below the deepest resolved level the model says nothing: fade to abyss.
      const f = Math.min(1, (t - LEVEL_SPAN) / (1 - LEVEL_SPAN || 1));
      ctx.fillStyle = "#0a1330";
      ctx.globalAlpha = 1 - f * 0.55;
    }
    ctx.fillRect(0, y, W, 1);
  }
  ctx.globalAlpha = 1;
}

function particleSprite(): THREE.CanvasTexture {
  const cv = document.createElement("canvas");
  cv.width = 64;
  cv.height = 64;
  const ctx = cv.getContext("2d");
  if (ctx) {
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.35, "rgba(190,235,255,0.7)");
    g.addColorStop(1, "rgba(190,235,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 64, 64);
  }
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Glowing ring + translucent sheet marking one depth. */
function makeIndicator(w: number, d: number, color: number, sheetOpacity: number): THREE.Group {
  const g = new THREE.Group();
  const half = Math.max(w, d) * 0.5;
  const ring = new THREE.RingGeometry(half * 1.01, half * 1.05, 4);
  ring.rotateX(-Math.PI / 2);
  ring.rotateY(Math.PI / 4);
  g.add(
    new THREE.Mesh(
      ring,
      new THREE.MeshBasicMaterial({
        color,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
      })
    )
  );
  const sheet = new THREE.PlaneGeometry(w, d);
  sheet.rotateX(-Math.PI / 2);
  g.add(
    new THREE.Mesh(
      sheet,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: sheetOpacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    )
  );
  return g;
}

export default function Volume3D({
  pred,
  depths,
  controls,
  picked,
  column,
  scale,
  thermocline,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  const basinGroupRef = useRef<THREE.Group | null>(null);
  const blockGroupRef = useRef<THREE.Group | null>(null);
  const sliceGroupRef = useRef<THREE.Group | null>(null);
  const curtainGroupRef = useRef<THREE.Group | null>(null);
  const strataRef = useRef<THREE.Group | null>(null);
  const particlesRef = useRef<THREE.Points | null>(null);
  const wallMatRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const capMatRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const gradCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const gradTexRef = useRef<THREE.CanvasTexture | null>(null);
  const blockCursorRef = useRef<THREE.Group | null>(null);
  const basinCursorRef = useRef<THREE.Group | null>(null);
  const thermoRef = useRef<THREE.Group | null>(null);
  const probeRef = useRef<THREE.Group | null>(null);

  const controlsRef = useRef(controls);
  const targetsRef = useRef({ depthT: 0, thermoT: -1 });
  // Textures are not reachable from a scene traverse, so they are tracked by
  // hand. Slice and curtain layers are rebuilt often and own their own lists.
  const disposables = useRef<Disposable[]>([]);
  const sliceJunk = useRef<Disposable[]>([]);
  const curtainJunk = useRef<Disposable[]>([]);

  const [hint, setHint] = useState(true);

  useEffect(() => {
    controlsRef.current = controls;
  }, [controls]);

  // Each basin level is scaled to its own range - the same convention the 2-D
  // dashboard uses when you sweep the depth slider. The column block instead
  // uses one scale across the whole series, so colour means the same thing on
  // every frame.
  const perDepthScale = useMemo(() => pred.truth.map((g) => gridExtent(g)), [pred]);

  const nD = depths.length;

  // ---- scene bootstrap (once) ------------------------------------------------
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 800;
    const height = mount.clientHeight || 560;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070d);
    scene.fog = new THREE.FogExp2(0x05070d, 0.0038);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    scene.add(new THREE.AmbientLight(0xdceeff, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(60, 120, 80);
    scene.add(key);

    // ---- basin scene ---------------------------------------------------------
    const basin = new THREE.Group();
    scene.add(basin);
    basinGroupRef.current = basin;

    basin.add(
      new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(BOX_X, BOX_Y, BOX_Z)),
        new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.28 })
      )
    );
    const surfaceRing = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(BOX_X, 0.001, BOX_Z)),
      new THREE.LineBasicMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.55 })
    );
    surfaceRing.position.y = BOX_Y * 0.5;
    basin.add(surfaceRing);

    const sliceGroup = new THREE.Group();
    basin.add(sliceGroup);
    sliceGroupRef.current = sliceGroup;

    const curtainGroup = new THREE.Group();
    basin.add(curtainGroup);
    curtainGroupRef.current = curtainGroup;

    const basinCursor = makeIndicator(BOX_X, BOX_Z, 0xf59e0b, 0.07);
    basin.add(basinCursor);
    basinCursorRef.current = basinCursor;

    const probe = new THREE.Group();
    probe.visible = false;
    probe.add(
      new THREE.Mesh(
        new THREE.CylinderGeometry(0.22, 0.22, BOX_Y, 8),
        new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.9 })
      )
    );
    const probeHead = new THREE.Mesh(
      new THREE.SphereGeometry(1.1, 20, 20),
      new THREE.MeshStandardMaterial({
        color: 0xfbbf24,
        emissive: 0xf59e0b,
        emissiveIntensity: 0.7,
      })
    );
    probeHead.position.y = BOX_Y * 0.5;
    probe.add(probeHead);
    basin.add(probe);
    probeRef.current = probe;

    // ---- volumetric water block ---------------------------------------------
    const block = new THREE.Group();
    scene.add(block);
    blockGroupRef.current = block;

    const gradCanvas = document.createElement("canvas");
    gradCanvas.width = 4;
    gradCanvas.height = GRAD_H;
    gradCanvasRef.current = gradCanvas;
    const gradTex = texFrom(gradCanvas);
    gradTexRef.current = gradTex;

    const wallMat = new THREE.MeshBasicMaterial({
      map: gradTex,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    wallMatRef.current = wallMat;

    // [planeWidth, x, z, rotationY]
    const walls: [number, number, number, number][] = [
      [BLK_X, 0, BLK_Z * 0.5, 0],
      [BLK_X, 0, -BLK_Z * 0.5, 0],
      [BLK_Z, BLK_X * 0.5, 0, Math.PI / 2],
      [BLK_Z, -BLK_X * 0.5, 0, Math.PI / 2],
    ];
    for (const [w, x, z, ry] of walls) {
      const geo = new THREE.PlaneGeometry(w, BLK_Y);
      const mesh = new THREE.Mesh(geo, wallMat);
      mesh.position.set(x, 0, z);
      mesh.rotation.y = ry;
      block.add(mesh);
    }

    // Sea-surface cap, tinted with the surface temperature of the live frame.
    const capGeo = new THREE.PlaneGeometry(BLK_X, BLK_Z);
    capGeo.rotateX(-Math.PI / 2);
    const capMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.4,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    capMatRef.current = capMat;
    const cap = new THREE.Mesh(capGeo, capMat);
    cap.position.y = BLK_Y * 0.5;
    block.add(cap);
    disposables.current.push(gradTex);

    block.add(
      new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(BLK_X, BLK_Y, BLK_Z)),
        new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.4 })
      )
    );

    // Faint boundary rings between resolved strata, plus a brighter one at the
    // deepest level the model actually resolves.
    const strata = new THREE.Group();
    const strataGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(BLK_X, 0.001, BLK_Z));
    const strataMat = new THREE.LineBasicMaterial({
      color: 0x7dd3fc,
      transparent: true,
      opacity: 0.16,
    });
    for (let i = 0; i < nD; i++) {
      const ring = new THREE.LineSegments(strataGeo, strataMat);
      ring.position.y = yFromT(levelT(i, nD), BLK_Y);
      strata.add(ring);
    }
    const floorMat = new THREE.LineBasicMaterial({
      color: 0x94a3b8,
      transparent: true,
      opacity: 0.45,
    });
    const floorRing = new THREE.LineSegments(strataGeo, floorMat);
    floorRing.position.y = yFromT(LEVEL_SPAN, BLK_Y);
    strata.add(floorRing);
    block.add(strata);
    strataRef.current = strata;

    // Ambient marine snow drifting through the column.
    const pGeo = new THREE.BufferGeometry();
    const pos = new Float32Array(PARTICLES * 3);
    const col = new Float32Array(PARTICLES * 3);
    const speed = new Float32Array(PARTICLES);
    for (let i = 0; i < PARTICLES; i++) {
      pos[i * 3] = (Math.random() - 0.5) * BLK_X * 0.94;
      pos[i * 3 + 1] = (Math.random() - 0.5) * BLK_Y * 0.98;
      pos[i * 3 + 2] = (Math.random() - 0.5) * BLK_Z * 0.94;
      const shallow = 0.45 + (pos[i * 3 + 1] / BLK_Y + 0.5) * 0.55;
      col[i * 3] = 0.55 * shallow;
      col[i * 3 + 1] = 0.85 * shallow;
      col[i * 3 + 2] = shallow;
      speed[i] = 0.6 + Math.random() * 2.2;
    }
    pGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    pGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const sprite = particleSprite();
    const pMat = new THREE.PointsMaterial({
      size: 0.95,
      map: sprite,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(pGeo, pMat);
    points.userData.speed = speed;
    block.add(points);
    particlesRef.current = points;
    disposables.current.push(sprite);

    const blockCursor = makeIndicator(BLK_X, BLK_Z, 0xf59e0b, 0.1);
    blockCursor.renderOrder = 30;
    block.add(blockCursor);
    blockCursorRef.current = blockCursor;

    // Thermocline: a cyan ring that tracks the steepest-gradient depth.
    const thermo = new THREE.Group();
    const thermoGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(BLK_X * 1.02, 0.001, BLK_Z * 1.02));
    thermo.add(
      new THREE.LineSegments(
        thermoGeo,
        new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.9 })
      )
    );
    thermo.visible = false;
    block.add(thermo);
    thermoRef.current = thermo;

    // ---- orbit controls (hand-rolled; no addon import needed) ----------------
    let dragging = false;
    let prev = { x: 0, y: 0 };
    const sph = { radius: 165, theta: 0.62, phi: 1.12 };
    let wantRadius = 165;

    const onDown = (e: PointerEvent) => {
      dragging = true;
      prev = { x: e.clientX, y: e.clientY };
      setHint(false);
      renderer.domElement.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      sph.theta -= (e.clientX - prev.x) * 0.006;
      sph.phi = Math.max(0.12, Math.min(Math.PI - 0.12, sph.phi - (e.clientY - prev.y) * 0.006));
      prev = { x: e.clientX, y: e.clientY };
    };
    const onUp = (e: PointerEvent) => {
      dragging = false;
      if (renderer.domElement.hasPointerCapture(e.pointerId)) {
        renderer.domElement.releasePointerCapture(e.pointerId);
      }
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      wantRadius = Math.max(70, Math.min(320, wantRadius + e.deltaY * 0.12));
      sph.radius = wantRadius;
    };

    const el = renderer.domElement;
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });

    const t0 = performance.now();
    let last = t0;
    let raf = 0;
    let lastMode: VolumeMode | null = null;

    const animate = () => {
      raf = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const t = (now - t0) / 1000;
      const c = controlsRef.current;

      if (lastMode !== c.mode) {
        lastMode = c.mode;
        wantRadius = c.mode === "column" ? 132 : 128;
      }
      sph.radius += (wantRadius - sph.radius) * 0.08;

      if (c.spin && !dragging) sph.theta += 0.0022;

      camera.position.set(
        sph.radius * Math.sin(sph.phi) * Math.sin(sph.theta),
        sph.radius * Math.cos(sph.phi),
        sph.radius * Math.sin(sph.phi) * Math.cos(sph.theta)
      );
      camera.lookAt(0, 0, 0);

      const inColumn = c.mode === "column";
      if (basinGroupRef.current) basinGroupRef.current.visible = !inColumn;
      if (blockGroupRef.current) blockGroupRef.current.visible = inColumn;

      const pulse = 0.55 + Math.sin(t * 3) * 0.3;
      const applyCursor = (g: THREE.Group | null, height: number, visible: boolean) => {
        if (!g) return;
        g.visible = visible;
        const target = yFromT(targetsRef.current.depthT, height);
        g.position.y += (target - g.position.y) * 0.18;
        const ring = g.children[0] as THREE.Mesh;
        (ring.material as THREE.MeshBasicMaterial).opacity = pulse;
      };
      applyCursor(blockCursorRef.current, BLK_Y, c.showCursor && inColumn);
      applyCursor(basinCursorRef.current, BOX_Y, c.showCursor && !inColumn);

      if (thermoRef.current) {
        const tt = targetsRef.current.thermoT;
        thermoRef.current.visible = tt >= 0 && inColumn;
        if (tt >= 0) {
          const target = yFromT(tt, BLK_Y);
          thermoRef.current.position.y += (target - thermoRef.current.position.y) * 0.14;
        }
      }

      const pts = particlesRef.current;
      if (pts) {
        pts.visible = c.showParticles && inColumn;
        if (pts.visible) {
          const attr = pts.geometry.getAttribute("position") as THREE.BufferAttribute;
          const arr = attr.array as Float32Array;
          const sp = pts.userData.speed as Float32Array;
          const top = BLK_Y * 0.5;
          for (let i = 0; i < PARTICLES; i++) {
            const yi = i * 3 + 1;
            arr[yi] += sp[i] * dt;
            if (arr[yi] > top) arr[yi] = -top;
            arr[i * 3] += Math.sin(t * 0.35 + i) * dt * 0.35;
          }
          attr.needsUpdate = true;
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      el.removeEventListener("wheel", onWheel);
      disposeTree(scene);
      purge(disposables);
      purge(sliceJunk);
      purge(curtainJunk);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
    // Built once; data and controls are pushed in by the effects below.
  }, [nD]);

  // ---- live targets read by the animation loop -------------------------------
  useEffect(() => {
    targetsRef.current.depthT = depthT(controls.depthM, depths);
  }, [controls.depthM, depths]);

  useEffect(() => {
    targetsRef.current.thermoT = thermocline === null ? -1 : depthT(thermocline, depths);
  }, [thermocline, depths]);

  // ---- block gradient: repainted per frame, never rebuilt ---------------------
  useEffect(() => {
    const cv = gradCanvasRef.current;
    const tex = gradTexRef.current;
    if (!cv || !tex || !column) return;
    paintGradient(cv, column.predicted, depths, scale.min, scale.max);
    tex.needsUpdate = true;

    const surface = column.predicted.find((v) => v !== null);
    if (capMatRef.current && surface !== undefined && surface !== null) {
      const [r, g, b] = thermalColor(surface, scale.min, scale.max);
      capMatRef.current.color.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    }
  }, [column, depths, scale.min, scale.max]);

  useEffect(() => {
    if (wallMatRef.current) wallMatRef.current.opacity = controls.opacity;
  }, [controls.opacity]);

  useEffect(() => {
    if (strataRef.current) strataRef.current.visible = controls.showStrata;
  }, [controls.showStrata]);

  // ---- depth-level stack -----------------------------------------------------
  useEffect(() => {
    const group = sliceGroupRef.current;
    if (!group) return;

    group.clear();
    const stride = Math.max(1, controls.sliceStride);
    const alpha = Math.round(255 * controls.opacity);

    for (let d = 0; d < nD; d += stride) {
      const { min, max } = perDepthScale[d];
      const tex = texFrom(levelCanvas(pred.predicted[d], min, max, alpha));
      const geo = new THREE.PlaneGeometry(BOX_X, BOX_Z);
      geo.rotateX(-Math.PI / 2);
      const mat = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.y = yFromT(levelT(d, nD), BOX_Y);
      mesh.renderOrder = d;
      group.add(mesh);
      sliceJunk.current.push(tex, geo, mat);
    }
    group.visible = controls.showSlices;
    return () => purge(sliceJunk);
  }, [pred, nD, perDepthScale, controls.sliceStride, controls.opacity, controls.showSlices]);

  // ---- cross-section curtains through the picked point ------------------------
  useEffect(() => {
    const group = curtainGroupRef.current;
    if (!group) return;
    group.clear();
    group.visible = controls.showCurtains;
    if (!controls.showCurtains || !picked) return;

    // A single scale across the whole column, so the thermocline reads as one
    // continuous gradient rather than 15 independently-stretched bands.
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of perDepthScale) {
      lo = Math.min(lo, s.min);
      hi = Math.max(hi, s.max);
    }

    const nLat = pred.lat.length;
    const nLon = pred.lon.length;
    // The resolved levels only occupy the top LEVEL_SPAN of the box.
    const curtainH = BOX_Y * LEVEL_SPAN;
    const curtainY = BOX_Y * 0.5 - curtainH * 0.5;

    const latTex = texFrom(latCurtainCanvas(pred.predicted, picked.row, lo, hi));
    const latGeo = new THREE.PlaneGeometry(BOX_X, curtainH);
    const latMat = new THREE.MeshBasicMaterial({
      map: latTex,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const latMesh = new THREE.Mesh(latGeo, latMat);
    latMesh.position.set(0, curtainY, BOX_Z * (0.5 - picked.row / (nLat - 1)) * -1);
    latMesh.renderOrder = 100;
    group.add(latMesh);

    const lonTex = texFrom(lonCurtainCanvas(pred.predicted, picked.col, lo, hi));
    const lonGeo = new THREE.PlaneGeometry(BOX_Z, curtainH);
    lonGeo.rotateY(Math.PI / 2);
    const lonMat = new THREE.MeshBasicMaterial({
      map: lonTex,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const lonMesh = new THREE.Mesh(lonGeo, lonMat);
    lonMesh.position.set(BOX_X * (picked.col / (nLon - 1) - 0.5), curtainY, 0);
    lonMesh.renderOrder = 101;
    group.add(lonMesh);

    curtainJunk.current.push(latTex, latGeo, latMat, lonTex, lonGeo, lonMat);
    return () => purge(curtainJunk);
  }, [pred, picked, perDepthScale, controls.showCurtains]);

  // ---- probe position --------------------------------------------------------
  useEffect(() => {
    const probe = probeRef.current;
    if (!probe) return;
    if (!picked) {
      probe.visible = false;
      return;
    }
    probe.visible = true;
    probe.position.x = BOX_X * (picked.col / (pred.lon.length - 1) - 0.5);
    probe.position.z = BOX_Z * (0.5 - picked.row / (pred.lat.length - 1)) * -1;
  }, [picked, pred.lat.length, pred.lon.length]);

  return (
    <div className="relative h-full w-full">
      <div ref={mountRef} className="h-full w-full cursor-grab active:cursor-grabbing" />

      {controls.mode === "column" && (
        <div className="pointer-events-none absolute right-3 top-3 z-10 rounded-lg border border-white/10 bg-slate-950/70 px-2.5 py-1.5 text-right backdrop-blur-md">
          <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">
            Column
          </div>
          <div className="font-mono text-[11px] text-slate-300">
            0 &ndash; {DEPTH_MAP_MAX} m
          </div>
          <div className="font-mono text-[10px] text-slate-500">
            {nD} resolved levels
          </div>
        </div>
      )}

      {hint && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-full border border-white/10 bg-slate-950/80 px-3 py-1.5 text-[11px] text-slate-400 backdrop-blur-md">
          Drag to orbit &middot; scroll to zoom
        </div>
      )}
    </div>
  );
}
