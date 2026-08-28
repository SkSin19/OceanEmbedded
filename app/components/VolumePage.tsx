"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import type { Meta, ModelName, Prediction } from "../lib/types";
import { gridExtent, nearestOcean } from "../lib/grid";
import type { VolumeControls } from "./Volume3D";
import ProfileChart from "./ProfileChart";
import Legend from "./Legend";

// WebGL only exists in the browser, so keep the scene out of the server render.
const Volume3D = dynamic(() => import("./Volume3D"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-sm text-slate-400">
      <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />
      Building volume...
    </div>
  ),
});

interface Props {
  meta: Meta;
  pred: Prediction | null;
  model: ModelName;
  loading: boolean;
  picked: { row: number; col: number } | null;
  onPick: (p: { row: number; col: number }) => void;
}

export default function VolumePage({ meta, pred, loading, picked, onPick }: Props) {
  const [controls, setControls] = useState<VolumeControls>({
    depthIdx: 5,
    showSlices: true,
    showCurtains: true,
    showCursor: true,
    spin: true,
    sliceStride: 1,
    opacity: 0.55,
  });

  // Default the probe to an ocean point in the Bay of Bengal.
  useEffect(() => {
    if (picked || !pred) return;
    onPick(nearestOcean(pred.truth[0], pred.lat, pred.lon, 15, 88));
  }, [pred, picked, onPick]);

  const set = <K extends keyof VolumeControls>(k: K, v: VolumeControls[K]) =>
    setControls((c) => ({ ...c, [k]: v }));

  const profile = useMemo(() => {
    if (!pred || !picked) return null;
    return {
      predicted: pred.predicted.map((d) => d[picked.row][picked.col]),
      truth: pred.truth.map((d) => d[picked.row][picked.col]),
      lat: pred.lat[picked.row],
      lon: pred.lon[picked.col],
    };
  }, [pred, picked]);

  // Column-wide colour scale, matching what the curtains are drawn with.
  const columnScale = useMemo(() => {
    if (!pred) return { min: 0, max: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (const g of pred.truth) {
      const e = gridExtent(g);
      min = Math.min(min, e.min);
      max = Math.max(max, e.max);
    }
    return { min, max };
  }, [pred]);

  if (!pred) {
    return (
      <div className="flex h-[60vh] items-center justify-center text-slate-400">
        <span className="mr-2 h-5 w-5 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />
        Loading reconstruction volume...
      </div>
    );
  }

  const mld = estimateThermocline(profile?.predicted ?? [], meta.depths);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="animate-fade-up mb-6 max-w-2xl">
        <div className="mb-2 inline-flex items-center gap-1.5 rounded-full bg-violet-500/10 px-2.5 py-1 text-[11px] font-medium tracking-wide text-violet-300 ring-1 ring-violet-500/20">
          Volumetric view
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50 sm:text-3xl">
          The reconstructed water column
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          All fifteen predicted depth levels stacked in space, plus two vertical
          cross-sections cut through the probe. Every texel is model output for{" "}
          <span className="font-mono text-slate-300">{pred.date}</span> — nothing
          here is an analytic stand-in.
        </p>
      </div>

      <div className="animate-fade-up grid grid-cols-1 gap-6 lg:grid-cols-3" style={{ animationDelay: "80ms" }}>
        <section className="glass-panel relative overflow-hidden rounded-2xl lg:col-span-2">
          <div className="h-[600px] w-full">
            <Volume3D pred={pred} meta={meta} controls={controls} picked={picked} />
          </div>

          {loading && (
            <div className="pointer-events-none absolute right-3 top-3 z-10 flex items-center gap-1.5 rounded-full border border-white/10 bg-slate-950/85 px-2.5 py-1 text-[11px] text-amber-300 backdrop-blur-md">
              <span className="h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-amber-900 border-t-amber-400" />
              Reconstructing
            </div>
          )}

          <div className="pointer-events-none absolute bottom-3 right-3 z-10 w-48 rounded-xl border border-white/10 bg-slate-950/85 p-2.5 backdrop-blur-md">
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="text-[11px] font-medium text-slate-200">Temperature</span>
              <span className="text-[10px] text-slate-500">degC</span>
            </div>
            <Legend mode="thermal" min={columnScale.min} max={columnScale.max} absMax={1} />
          </div>
        </section>

        <section className="flex flex-col gap-4">
          <div className="glass-panel rounded-2xl p-4">
            <h2 className="mb-3 text-sm font-medium text-slate-200">Scene</h2>

            <div className="space-y-2">
              <Toggle
                label="Depth-level stack"
                checked={controls.showSlices}
                onChange={(v) => set("showSlices", v)}
              />
              <Toggle
                label="Cross-section curtains"
                checked={controls.showCurtains}
                onChange={(v) => set("showCurtains", v)}
              />
              <Toggle
                label="Depth cursor"
                checked={controls.showCursor}
                onChange={(v) => set("showCursor", v)}
              />
              <Toggle label="Auto-rotate" checked={controls.spin} onChange={(v) => set("spin", v)} />
            </div>

            <div className="mt-4 space-y-3">
              <div>
                <label className="flex items-center justify-between text-xs text-slate-300">
                  <span>Cursor depth</span>
                  <span className="font-mono text-amber-400">
                    {meta.depths[controls.depthIdx]} m
                  </span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={meta.depths.length - 1}
                  value={controls.depthIdx}
                  onChange={(e) => set("depthIdx", Number(e.target.value))}
                  className="mt-1.5 w-full accent-amber-500"
                />
              </div>
              <div>
                <label className="flex items-center justify-between text-xs text-slate-300">
                  <span>Slice opacity</span>
                  <span className="font-mono text-cyan-400">
                    {Math.round(controls.opacity * 100)}%
                  </span>
                </label>
                <input
                  type="range"
                  min={0.15}
                  max={1}
                  step={0.05}
                  value={controls.opacity}
                  onChange={(e) => set("opacity", Number(e.target.value))}
                  className="mt-1.5 w-full accent-cyan-400"
                />
              </div>
              <div>
                <label className="flex items-center justify-between text-xs text-slate-300">
                  <span>Levels drawn</span>
                  <span className="font-mono text-cyan-400">
                    {Math.ceil(meta.depths.length / controls.sliceStride)} / {meta.depths.length}
                  </span>
                </label>
                <input
                  type="range"
                  min={1}
                  max={5}
                  step={1}
                  value={controls.sliceStride}
                  onChange={(e) => set("sliceStride", Number(e.target.value))}
                  className="mt-1.5 w-full accent-cyan-400"
                />
              </div>
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-4">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-medium text-slate-200">Probe column</h2>
              {profile && (
                <span className="rounded-md bg-slate-800/80 px-1.5 py-0.5 font-mono text-[11px] text-slate-400">
                  {profile.lat.toFixed(2)} N, {profile.lon.toFixed(2)} E
                </span>
              )}
            </div>

            {mld && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                <span className="rounded-md bg-cyan-500/10 px-1.5 py-0.5 font-mono text-[11px] text-cyan-300">
                  Thermocline ~{mld.depth} m
                </span>
                <span className="rounded-md bg-slate-800/80 px-1.5 py-0.5 font-mono text-[11px] text-slate-400">
                  {mld.gradient.toFixed(3)} degC/m
                </span>
              </div>
            )}

            <ProfileChart
              depths={meta.depths}
              predicted={profile?.predicted ?? []}
              truth={profile?.truth ?? []}
            />
            <div className="mt-2 flex gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 bg-sky-400" /> Truth (GLORYS)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 bg-amber-500" /> Predicted
              </span>
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
              Move the probe from the Map tab — the curtains re-cut through
              whichever point you select there.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

/** Steepest temperature gradient in the profile: a first-order thermocline depth. */
function estimateThermocline(
  profile: (number | null)[],
  depths: number[]
): { depth: number; gradient: number } | null {
  let best: { depth: number; gradient: number } | null = null;
  for (let i = 1; i < profile.length; i++) {
    const a = profile[i - 1];
    const b = profile[i];
    if (a === null || b === null) continue;
    const dz = depths[i] - depths[i - 1];
    if (dz <= 0) continue;
    const g = (b - a) / dz;
    if (!best || g < best.gradient) {
      best = { depth: Math.round((depths[i] + depths[i - 1]) / 2), gradient: g };
    }
  }
  return best;
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between rounded-lg px-1 py-1 text-xs text-slate-300 transition hover:text-white"
    >
      <span>{label}</span>
      <span
        className={`relative h-4 w-7 rounded-full transition-colors ${
          checked ? "bg-cyan-500/70" : "bg-slate-700"
        }`}
      >
        <span
          className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
            checked ? "translate-x-3.5" : "translate-x-0.5"
          }`}
        />
      </span>
    </button>
  );
}
