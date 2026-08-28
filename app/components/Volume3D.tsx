"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Grid, Meta, Prediction } from "../lib/types";
import { gridExtent } from "../lib/grid";
import { thermalColor } from "../lib/color";

// Scene proportions. X spans longitude, Z spans latitude, Y is the water column.
const BOX_X = 104;
const BOX_Z = 44;
const BOX_Y = 54;

export interface VolumeControls {
  /** Depth index highlighted by the cursor plane. */
  depthIdx: number;
  /** Show the horizontal stack of depth-level slices. */
  showSlices: boolean;
  /** Show the two vertical cross-section curtains through the picked point. */
  showCurtains: boolean;
  /** Show the isosurface-style contour on the cursor plane. */
  showCursor: boolean;
  /** Auto-rotate the camera. */
  spin: boolean;
  /** How many of the 15 levels to draw in the stack (perf / clarity). */
  sliceStride: number;
  /** Overall slice opacity. */
  opacity: number;
}

interface Props {
  pred: Prediction;
  meta: Meta;
  controls: VolumeControls;
  picked: { row: number; col: number } | null;
}

/** Even spacing by depth index: the levels are already log-ish, and this keeps
 *  the near-surface thermocline legible instead of squashed into a sliver. */
function depthToY(i: number, n: number): number {
  return BOX_Y * 0.5 - (i / (n - 1)) * BOX_Y;
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
function latCurtainCanvas(
  volume: Grid[],
  row: number,
  min: number,
  max: number
): HTMLCanvasElement {
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
function lonCurtainCanvas(
  volume: Grid[],
  col: number,
  min: number,
  max: number
): HTMLCanvasElement {
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

export default function Volume3D({ pred, meta, controls, picked }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  const sliceGroupRef = useRef<THREE.Group | null>(null);
  const curtainGroupRef = useRef<THREE.Group | null>(null);
  const cursorRef = useRef<THREE.Group | null>(null);
  const probeRef = useRef<THREE.Group | null>(null);
  const controlsRef = useRef(controls);
  const disposables = useRef<{ dispose: () => void }[]>([]);

  const [hint, setHint] = useState(true);

  useEffect(() => {
    controlsRef.current = controls;
  }, [controls]);

  // One shared colour scale across all depths would wash out the deep levels,
  // so each level is scaled to its own range - the same convention the 2-D
  // dashboard uses when you sweep the depth slider.
  const perDepthScale = useMemo(
    () => pred.truth.map((g) => gridExtent(g)),
    [pred]
  );

  const nD = pred.depths.length;

  // ---- scene bootstrap (once) ------------------------------------------------
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 800;
    const height = mount.clientHeight || 560;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070d);
    scene.fog = new THREE.FogExp2(0x05070d, 0.0042);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    cameraRef.current = camera;

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

    // Domain wireframe + a faint sea-surface plane for orientation.
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(BOX_X, BOX_Y, BOX_Z)),
      new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.28 })
    );
    scene.add(box);

    const surfaceRing = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(BOX_X, 0.001, BOX_Z)),
      new THREE.LineBasicMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.55 })
    );
    surfaceRing.position.y = BOX_Y * 0.5;
    scene.add(surfaceRing);

    const sliceGroup = new THREE.Group();
    scene.add(sliceGroup);
    sliceGroupRef.current = sliceGroup;

    const curtainGroup = new THREE.Group();
    scene.add(curtainGroup);
    curtainGroupRef.current = curtainGroup;

    // Depth cursor: a glowing ring + translucent sheet at the active level.
    const cursor = new THREE.Group();
    const ringGeo = new THREE.RingGeometry(BOX_X * 0.5 * 1.01, BOX_X * 0.5 * 1.04, 4);
    ringGeo.rotateX(-Math.PI / 2);
    cursor.add(
      new THREE.Mesh(
        ringGeo,
        new THREE.MeshBasicMaterial({
          color: 0xf59e0b,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        })
      )
    );
    const sheetGeo = new THREE.PlaneGeometry(BOX_X, BOX_Z);
    sheetGeo.rotateX(-Math.PI / 2);
    cursor.add(
      new THREE.Mesh(
        sheetGeo,
        new THREE.MeshBasicMaterial({
          color: 0xf59e0b,
          transparent: true,
          opacity: 0.07,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
      )
    );
    scene.add(cursor);
    cursorRef.current = cursor;

    // Probe: a vertical line + head marking the picked water column.
    const probe = new THREE.Group();
    probe.visible = false;
    const probeLine = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 0.22, BOX_Y, 8),
      new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.9 })
    );
    probe.add(probeLine);
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
    scene.add(probe);
    probeRef.current = probe;

    // ---- orbit controls (hand-rolled; no addon import needed) ----------------
    let dragging = false;
    let prev = { x: 0, y: 0 };
    const sph = { radius: 165, theta: 0.62, phi: 1.12 };

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
      sph.radius = Math.max(70, Math.min(320, sph.radius + e.deltaY * 0.12));
    };

    const el = renderer.domElement;
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });

    const t0 = performance.now();
    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      const t = (performance.now() - t0) / 1000;
      const c = controlsRef.current;

      if (c.spin && !dragging) sph.theta += 0.0022;

      camera.position.set(
        sph.radius * Math.sin(sph.phi) * Math.sin(sph.theta),
        sph.radius * Math.cos(sph.phi),
        sph.radius * Math.sin(sph.phi) * Math.cos(sph.theta)
      );
      camera.lookAt(0, 0, 0);

      if (cursorRef.current) {
        cursorRef.current.visible = c.showCursor;
        const targetY = depthToY(c.depthIdx, nD);
        cursorRef.current.position.y += (targetY - cursorRef.current.position.y) * 0.16;
        const pulse = 0.6 + Math.sin(t * 3) * 0.25;
        const ring = cursorRef.current.children[0] as THREE.Mesh;
        (ring.material as THREE.MeshBasicMaterial).opacity = pulse;
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
      disposables.current.forEach((d) => d.dispose());
      disposables.current = [];
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
    // Built once; data and controls are pushed in by the effects below.
  }, [nD]);

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
        opacity: 1,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.y = depthToY(d, nD);
      mesh.renderOrder = d;
      group.add(mesh);
      disposables.current.push(tex, geo, mat);
    }
    group.visible = controls.showSlices;
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

    // Constant-latitude curtain: spans longitude.
    const latTex = texFrom(latCurtainCanvas(pred.predicted, picked.row, lo, hi));
    const latGeo = new THREE.PlaneGeometry(BOX_X, BOX_Y);
    const latMat = new THREE.MeshBasicMaterial({
      map: latTex,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const latMesh = new THREE.Mesh(latGeo, latMat);
    latMesh.position.z = BOX_Z * (0.5 - picked.row / (nLat - 1)) * -1;
    latMesh.renderOrder = 100;
    group.add(latMesh);

    // Constant-longitude curtain: spans latitude.
    const lonTex = texFrom(lonCurtainCanvas(pred.predicted, picked.col, lo, hi));
    const lonGeo = new THREE.PlaneGeometry(BOX_Z, BOX_Y);
    lonGeo.rotateY(Math.PI / 2);
    const lonMat = new THREE.MeshBasicMaterial({
      map: lonTex,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const lonMesh = new THREE.Mesh(lonGeo, lonMat);
    lonMesh.position.x = BOX_X * (picked.col / (nLon - 1) - 0.5);
    lonMesh.renderOrder = 101;
    group.add(lonMesh);

    disposables.current.push(latTex, latGeo, latMat, lonTex, lonGeo, lonMat);
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

      {/* Depth rail: real level values against the vertical axis. */}
      <div className="pointer-events-none absolute left-3 top-1/2 z-10 -translate-y-1/2 space-y-0.5 rounded-lg border border-white/10 bg-slate-950/70 px-2 py-2 backdrop-blur-md">
        <div className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-slate-500">
          Depth
        </div>
        {meta.depths.map((d, i) => (
          <div
            key={d}
            className={`text-right font-mono text-[10px] leading-[1.15] transition-colors ${
              i === controls.depthIdx ? "font-bold text-amber-400" : "text-slate-500"
            }`}
          >
            {d}
          </div>
        ))}
        <div className="pt-0.5 text-right text-[9px] text-slate-600">m</div>
      </div>

      {hint && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-full border border-white/10 bg-slate-950/80 px-3 py-1.5 text-[11px] text-slate-400 backdrop-blur-md">
          Drag to orbit &middot; scroll to zoom
        </div>
      )}
    </div>
  );
}
