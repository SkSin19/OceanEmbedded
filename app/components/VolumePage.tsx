"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import type { Meta, ModelName, Prediction } from "../lib/types";
import type { ColumnFrame, Source } from "../lib/api";
import { loadColumnSeries, resampleTimes, seedProfileCache } from "../lib/api";
import { nearestOcean } from "../lib/grid";
import {
  DEPTH_MAP_MAX,
  estimateThermocline,
  nearestLevel,
  profileExtent,
  sampleProfile,
} from "../lib/depth";
import type { VolumeControls, VolumeMode } from "./Volume3D";
import DepthMap from "./DepthMap";
import TemporalProfileChart from "./TemporalProfileChart";
import Timeline from "./Timeline";
import DeltaCards from "./DeltaCards";
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

const MAX_FRAMES = 40;

interface Props {
  source: Source;
  meta: Meta;
  pred: Prediction | null;
  date: string;
  model: ModelName;
  loading: boolean;
  picked: { row: number; col: number } | null;
  onPick: (p: { row: number; col: number }) => void;
  onDate: (d: string) => void;
}

export default function VolumePage({
  source,
  meta,
  pred,
  date,
  model,
  loading,
  picked,
  onPick,
  onDate,
}: Props) {
  const [controls, setControls] = useState<VolumeControls>({
    mode: "column",
    depthIdx: 5,
    depthM: 50,
    showStrata: true,
    showParticles: true,
    showSlices: true,
    showCurtains: true,
    showCursor: true,
    spin: true,
    sliceStride: 1,
    opacity: 0.55,
  });

  const dates = useMemo(() => resampleTimes(meta.times, MAX_FRAMES), [meta.times]);
  // Keyed by the point being profiled: moving the probe invalidates the series
  // by key rather than by resetting state from an effect.
  const [series, setSeries] = useState<{ key: string | null; frames: (ColumnFrame | null)[] }>({
    key: null,
    frames: [],
  });
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(2);
  const [compare, setCompare] = useState(false);
  const [baseIndex, setBaseIndex] = useState(0);
  const [seriesError, setSeriesError] = useState<{ key: string; message: string } | null>(null);

  const lat = pred && picked ? pred.lat[picked.row] : null;
  const lon = pred && picked ? pred.lon[picked.col] : null;

  // Default the probe to an ocean point in the Bay of Bengal.
  useEffect(() => {
    if (picked || !pred) return;
    onPick(nearestOcean(pred.truth[0], pred.lat, pred.lon, 15, 88));
  }, [pred, picked, onPick]);

  const seriesKey = lat === null || lon === null ? null : `${source}|${model}|${lat}|${lon}`;

  // The grid already in memory covers one date for free.
  const localColumn = useMemo<ColumnFrame | null>(() => {
    if (!pred || !picked) return null;
    return {
      date: pred.date,
      predicted: pred.predicted.map((g) => g[picked.row][picked.col]),
      truth: pred.truth.map((g) => g[picked.row][picked.col]),
    };
  }, [pred, picked]);

  // Frames for the current point, with the in-memory column merged in. Reading
  // by key means a probe move invalidates the series without a reset effect.
  const frames = useMemo(() => {
    const base =
      series.key === seriesKey && series.frames.length === dates.length
        ? series.frames
        : dates.map(() => null);
    if (!localColumn) return base;
    const i = dates.indexOf(localColumn.date);
    if (i < 0 || base[i]) return base;
    const next = [...base];
    next[i] = localColumn;
    return next;
  }, [series, seriesKey, dates, localColumn]);

  // Warm the profile cache with the column we already have, so the series
  // loader below spends no request on it.
  useEffect(() => {
    if (!pred || !picked || !localColumn) return;
    seedProfileCache(
      {
        lat: pred.lat[picked.row],
        lon: pred.lon[picked.col],
        depths: meta.depths,
        predicted: localColumn.predicted,
        truth: localColumn.truth,
      },
      localColumn.date,
      model
    );
  }, [pred, picked, localColumn, meta.depths, model]);

  // One lightweight /profile call per date, instead of a grid per timestep.
  useEffect(() => {
    if (seriesKey === null || lat === null || lon === null || source !== "api") return;

    const ctrl = new AbortController();
    const slot = new Map(dates.map((d, i) => [d, i]));
    loadColumnSeries(dates, lat, lon, model, {
      signal: ctrl.signal,
      onFrame: (f) =>
        setSeries((prev) => {
          const i = slot.get(f.date);
          if (i === undefined) return prev;
          const base =
            prev.key === seriesKey && prev.frames.length === dates.length
              ? prev.frames
              : dates.map(() => null);
          const next = [...base];
          next[i] = f;
          return { key: seriesKey, frames: next };
        }),
    }).catch((e) => {
      if (!ctrl.signal.aborted) setSeriesError({ key: seriesKey, message: String(e) });
    });
    return () => ctrl.abort();
  }, [seriesKey, lat, lon, model, source, dates]);

  const ready = useMemo(() => frames.map((f) => f !== null), [frames]);
  const loadedCount = useMemo(() => frames.reduce((n, f) => n + (f ? 1 : 0), 0), [frames]);

  // While a frame is still in flight, hold the nearest one that has arrived so
  // the block never blanks out mid-playback.
  const current = useMemo(() => {
    if (frames[index]) return frames[index];
    for (let d = 1; d < frames.length; d++) {
      if (frames[index - d]) return frames[index - d];
      if (frames[index + d]) return frames[index + d];
    }
    return null;
  }, [frames, index]);

  const reference = useMemo(() => frames.find((f) => f !== null) ?? null, [frames]);

  // One scale across the whole series, so colour means the same on every frame.
  const scale = useMemo(() => {
    const all: (number | null)[][] = [];
    for (const f of frames) {
      if (!f) continue;
      all.push(f.predicted, f.truth);
    }
    return all.length ? profileExtent(all) : { min: 0, max: 30 };
  }, [frames]);

  const thermocline = useMemo(
    () => (current ? estimateThermocline(current.predicted, meta.depths) : null),
    [current, meta.depths]
  );

  const set = <K extends keyof VolumeControls>(k: K, v: VolumeControls[K]) =>
    setControls((c) => ({ ...c, [k]: v }));

  const setDepth = (m: number) =>
    setControls((c) => ({
      ...c,
      depthM: Math.max(0, Math.min(DEPTH_MAP_MAX, m)),
      depthIdx: nearestLevel(m, meta.depths),
    }));

  const atDepth = current
    ? {
        predicted: sampleProfile(current.predicted, meta.depths, controls.depthM),
        truth: sampleProfile(current.truth, meta.depths, controls.depthM),
      }
    : null;

  if (!pred) {
    return (
      <div className="flex h-[60vh] items-center justify-center text-slate-400">
        <span className="mr-2 h-5 w-5 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />
        Loading reconstruction volume...
      </div>
    );
  }

  const frameDate = dates[index] ?? date;
  const inSync = frameDate === pred.date;
  const belowFloor = controls.depthM > meta.depths[meta.depths.length - 1];

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="animate-fade-up mb-6 max-w-2xl">
        <div className="mb-2 inline-flex items-center gap-1.5 rounded-full bg-violet-500/10 px-2.5 py-1 text-[11px] font-medium tracking-wide text-violet-300 ring-1 ring-violet-500/20">
          Temporal depth explorer
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50 sm:text-3xl">
          How the water column changes
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          A volumetric block of ocean at{" "}
          <span className="font-mono text-slate-300">
            {lat?.toFixed(2)} N, {lon?.toFixed(2)} E
          </span>
          , coloured by predicted temperature from the surface down to the model
          floor. Scrub the timeline to watch the thermocline breathe; drag the
          depth map to cut the block at any level.
        </p>
      </div>

      <div className="animate-fade-up mb-4" style={{ animationDelay: "60ms" }}>
        <Timeline
          dates={dates}
          index={index}
          onIndex={setIndex}
          playing={playing}
          onPlaying={setPlaying}
          fps={fps}
          onFps={setFps}
          ready={ready}
          loaded={loadedCount}
        />
      </div>

      <div
        className="animate-fade-up grid grid-cols-1 gap-6 lg:grid-cols-3"
        style={{ animationDelay: "120ms" }}
      >
        <section className="glass-panel relative overflow-hidden rounded-2xl lg:col-span-2">
          <div className="h-120 w-full">
            <Volume3D
              pred={pred}
              depths={meta.depths}
              controls={controls}
              picked={picked}
              column={current}
              scale={scale}
              thermocline={thermocline?.depth ?? null}
            />
          </div>

          <div className="absolute left-3 top-1/2 z-10 -translate-y-1/2">
            <DepthMap
              depths={meta.depths}
              value={controls.depthM}
              onChange={setDepth}
              profile={current?.predicted ?? []}
              scale={scale}
              thermocline={thermocline?.depth ?? null}
            />
          </div>

          <div className="pointer-events-none absolute left-1/2 top-3 z-10 flex -translate-x-1/2 items-center gap-2">
            <span className="rounded-full border border-white/10 bg-slate-950/85 px-2.5 py-1 font-mono text-[11px] text-slate-300 backdrop-blur-md">
              {frameDate}
            </span>
            {atDepth?.predicted !== null && atDepth !== null && !belowFloor && (
              <span className="rounded-full border border-amber-500/25 bg-amber-500/10 px-2.5 py-1 font-mono text-[11px] text-amber-300 backdrop-blur-md">
                {atDepth.predicted?.toFixed(2)} degC @ {Math.round(controls.depthM)} m
              </span>
            )}
          </div>

          {loading && (
            <div className="pointer-events-none absolute right-3 top-16 z-10 flex items-center gap-1.5 rounded-full border border-white/10 bg-slate-950/85 px-2.5 py-1 text-[11px] text-amber-300 backdrop-blur-md">
              <span className="h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-amber-900 border-t-amber-400" />
              Reconstructing
            </div>
          )}

          <div className="pointer-events-none absolute bottom-3 right-3 z-10 w-48 rounded-xl border border-white/10 bg-slate-950/85 p-2.5 backdrop-blur-md">
            <div className="mb-1.5 flex items-baseline justify-between">
              <span className="text-[11px] font-medium text-slate-200">Temperature</span>
              <span className="text-[10px] text-slate-500">degC</span>
            </div>
            <Legend mode="thermal" min={scale.min} max={scale.max} absMax={1} />
          </div>
        </section>

        <section className="flex flex-col gap-4">
          <div className="glass-panel rounded-2xl p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-medium text-slate-200">Scene</h2>
              <div className="grid grid-cols-2 gap-0.5 rounded-lg bg-white/5 p-0.5 text-[11px] ring-1 ring-white/10">
                {(["column", "basin"] as VolumeMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => set("mode", m)}
                    className={`rounded-md px-3 py-1 text-center capitalize transition ${
                      controls.mode === m
                        ? "bg-gradient-to-b from-slate-600 to-slate-700 text-white"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              {controls.mode === "column" ? (
                <>
                  <Toggle
                    label="Depth strata rings"
                    checked={controls.showStrata}
                    onChange={(v) => set("showStrata", v)}
                  />
                  <Toggle
                    label="Marine snow"
                    checked={controls.showParticles}
                    onChange={(v) => set("showParticles", v)}
                  />
                </>
              ) : (
                <>
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
                </>
              )}
              <Toggle
                label="Depth indicator plane"
                checked={controls.showCursor}
                onChange={(v) => set("showCursor", v)}
              />
              <Toggle label="Auto-rotate" checked={controls.spin} onChange={(v) => set("spin", v)} />
            </div>

            <div className="mt-4 space-y-3">
              <div>
                <label className="flex items-center justify-between text-xs text-slate-300">
                  <span>{controls.mode === "column" ? "Wall opacity" : "Slice opacity"}</span>
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
              {controls.mode === "basin" && (
                <div>
                  <label className="flex items-center justify-between text-xs text-slate-300">
                    <span>Levels drawn</span>
                    <span className="font-mono text-cyan-400">
                      {Math.ceil(meta.depths.length / controls.sliceStride)} /{" "}
                      {meta.depths.length}
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
              )}
            </div>

            {controls.mode === "basin" && (
              <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
                The spatial layers are the full grid for{" "}
                <span className="font-mono text-slate-400">{pred.date}</span>.
                {!inSync && (
                  <>
                    {" "}
                    <button
                      onClick={() => onDate(frameDate)}
                      className="font-medium text-cyan-400 underline-offset-2 hover:underline"
                    >
                      Load {frameDate}
                    </button>{" "}
                    to match the timeline.
                  </>
                )}
              </p>
            )}
          </div>

          <div className="glass-panel rounded-2xl p-4">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-medium text-slate-200">Depth vs temperature</h2>
              <span className="rounded-md bg-slate-800/80 px-1.5 py-0.5 font-mono text-[11px] text-slate-400">
                {frameDate}
              </span>
            </div>

            {thermocline && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                <span className="rounded-md bg-cyan-500/10 px-1.5 py-0.5 font-mono text-[11px] text-cyan-300">
                  Thermocline ~{thermocline.depth} m
                </span>
                <span className="rounded-md bg-slate-800/80 px-1.5 py-0.5 font-mono text-[11px] text-slate-400">
                  {thermocline.gradient.toFixed(3)} degC/m
                </span>
              </div>
            )}

            <TemporalProfileChart
              depths={meta.depths}
              predicted={current?.predicted ?? []}
              truth={current?.truth ?? []}
              reference={reference && reference !== current ? reference.predicted : null}
              scale={scale}
              thermocline={thermocline?.depth ?? null}
              activeDepth={controls.depthM}
            />

            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 bg-sky-400" /> Truth (GLORYS)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 bg-amber-500" /> Predicted
              </span>
              {reference && reference !== current && (
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-4 bg-slate-600" /> {reference.date}
                </span>
              )}
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-medium text-slate-200">Compare frames</h2>
              <Toggle label="" checked={compare} onChange={setCompare} />
            </div>
            {compare ? (
              <DeltaCards
                depths={meta.depths}
                a={frames[baseIndex]}
                b={current}
                dates={dates}
                aIndex={baseIndex}
                onAIndex={setBaseIndex}
                ready={ready}
              />
            ) : (
              <p className="text-[11px] leading-relaxed text-slate-500">
                Turn this on to difference the live frame against any other day:
                thermocline shift, surface and column temperature deltas.
              </p>
            )}
          </div>

          {source !== "api" && (
            <p className="rounded-xl border border-amber-500/20 bg-amber-500/[0.06] px-3 py-2 text-[11px] leading-relaxed text-amber-200/80">
              Static sample data only covers{" "}
              <span className="font-mono">{pred.date}</span>. Start the FastAPI
              backend to scrub the full {meta.times.length}-day series.
            </p>
          )}
          {seriesError?.key === seriesKey && (
            <p className="rounded-xl border border-rose-500/20 bg-rose-500/[0.06] px-3 py-2 text-[11px] leading-relaxed text-rose-200/80">
              Time series incomplete: {seriesError.message}
            </p>
          )}
        </section>
      </div>
    </div>
  );
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
      aria-pressed={checked}
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
