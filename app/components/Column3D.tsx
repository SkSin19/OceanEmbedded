"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { thermalColor } from "../lib/color";

export interface Column {
  label: string;
  temps: (number | null)[];
}

interface Props {
  columns: Column[];
  depths: number[];
  min: number;
  max: number;
}

const COL_HEIGHT = 4.2;

/**
 * A rotatable 3D water column: each of the depth levels is a disk stacked by
 * depth, colored by temperature and widened when warmer, so the mixed layer,
 * thermocline, and cold deep water read at a glance. Multiple columns
 * (e.g. predicted vs truth) sit side by side.
 */
export default function Column3D({ columns, depths, min, max }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const dataRef = useRef({ columns, depths, min, max });
  dataRef.current = { columns, depths, min, max };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 1.4, 8.2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(4, 8, 6);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x5b8bff, 0.5);
    rim.position.set(-6, 2, -4);
    scene.add(rim);

    const root = new THREE.Group();
    scene.add(root);

    const disposables: Array<{ dispose: () => void }> = [];

    function build() {
      root.clear();
      const { columns: cols, depths: dep, min: lo, max: hi } = dataRef.current;
      const n = dep.length;
      const gap = cols.length > 1 ? 3.0 : 0;
      const startX = -gap * (cols.length - 1) / 2;

      cols.forEach((col, ci) => {
        const g = new THREE.Group();
        g.position.x = startX + ci * gap;
        for (let i = 0; i < n; i++) {
          const t = col.temps[i];
          if (t === null || Number.isNaN(t)) continue;
          const norm = Math.max(0, Math.min(1, (t - lo) / (hi - lo || 1)));
          const radius = 0.45 + norm * 0.95;
          const h = (COL_HEIGHT / n) * 0.82;
          const y = COL_HEIGHT / 2 - (i / (n - 1)) * COL_HEIGHT;
          const geo = new THREE.CylinderGeometry(radius, radius, h, 48);
          const [r, gc, b] = thermalColor(t, lo, hi);
          const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(r / 255, gc / 255, b / 255),
            roughness: 0.35,
            metalness: 0.15,
            emissive: new THREE.Color(r / 255, gc / 255, b / 255),
            emissiveIntensity: 0.12,
          });
          const disk = new THREE.Mesh(geo, mat);
          disk.position.y = y;
          g.add(disk);
          disposables.push(geo, mat);
        }
        root.add(g);
      });
    }
    build();

    // Interaction: auto-rotate, drag to spin, wheel to zoom.
    let velocity = 0.004;
    let dragging = false;
    let lastX = 0;
    let dist = 8.2;
    const onDown = (e: PointerEvent) => {
      dragging = true;
      lastX = e.clientX;
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      root.rotation.y += (e.clientX - lastX) * 0.01;
      lastX = e.clientX;
      velocity = 0.004;
    };
    const onUp = () => (dragging = false);
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      dist = Math.max(5, Math.min(13, dist + e.deltaY * 0.01));
    };
    const el = renderer.domElement;
    el.style.touchAction = "none";
    el.style.cursor = "grab";
    el.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });

    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      if (!dragging) root.rotation.y += velocity;
      camera.position.setLength(dist);
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    };
    animate();

    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight || 360;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    // Rebuild geometry when the incoming data changes.
    const rebuild = () => {
      disposables.splice(0).forEach((d) => d.dispose());
      build();
    };
    (mount as HTMLDivElement & { __rebuild?: () => void }).__rebuild = rebuild;

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      el.removeEventListener("wheel", onWheel);
      disposables.forEach((d) => d.dispose());
      renderer.dispose();
      if (el.parentNode === mount) mount.removeChild(el);
    };
  }, []);

  // Trigger a rebuild when inputs change (the effect stores a rebuild hook).
  useEffect(() => {
    const m = mountRef.current as (HTMLDivElement & { __rebuild?: () => void }) | null;
    m?.__rebuild?.();
  }, [columns, depths, min, max]);

  return (
    <div className="relative">
      <div ref={mountRef} className="h-[360px] w-full" />
      <div className="pointer-events-none absolute left-2 top-2 text-[10px] uppercase tracking-wide text-slate-500">
        drag to rotate . scroll to zoom
      </div>
      <div className="mt-1 flex justify-center gap-6 text-xs text-slate-400">
        {columns.map((c) => (
          <span key={c.label}>{c.label}</span>
        ))}
      </div>
    </div>
  );
}
