"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { Embedding, Grid, Meta, Metrics, ModelName, Prediction } from "../lib/types";
import { loadEmbedding, loadMetrics, type Source, type Validation } from "../lib/api";
import { errorGrid, gridExtent } from "../lib/grid";
import { MODEL_LABEL } from "../lib/labels";
import HeatmapCanvas from "./HeatmapCanvas";
import EmbeddingCanvas from "./EmbeddingCanvas";
import ProfileChart from "./ProfileChart";
import SkillChart from "./SkillChart";
import Legend from "./Legend";

type View = "predicted" | "truth" | "error" | "embedding";

interface DashboardProps {
  source: Source;
  meta: Meta;
  models: ModelName[];
  pred: Prediction | null;
  date: string;
  model: ModelName;
  loading: boolean;
  picked: { row: number; col: number } | null;
  onDate: (d: string) => void;
  onModel: (m: ModelName) => void;
  onPick: (p: { row: number; col: number }) => void;
}

export default function Dashboard({
  source,
  meta,
  models,
  pred,
  date,
  model,
  loading: switching,
  picked,
  onDate,
  onModel,
  onPick,
}: DashboardProps) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [depthIdx, setDepthIdx] = useState(0);
  const [view, setView] = useState<View>("predicted");
  const [embedding, setEmbedding] = useState<Embedding | null>(null);
  const [validation, setValidation] = useState<Validation>("holdout");

  // Metrics follow the selected model and validation source.
  useEffect(() => {
    let active = true;
    loadMetrics(source, model, validation)
      .then((k) => active && setMetrics(k))
      .catch((e) => active && setError(String(e)));
    return () => {
      active = false;
    };
  }, [source, model, validation]);

  // Embedding is loaded on demand when its view is active.
  useEffect(() => {
    if (view !== "embedding") return;
    let active = true;
    loadEmbedding(source, date)
      .then((e) => active && setEmbedding(e))
      .catch((e) => active && setError(String(e)));
    return () => {
      active = false;
    };
  }, [source, date, view]);

  const displayGrid = useMemo<Grid | null>(() => {
    if (!pred) return null;
    if (view === "error") return errorGrid(pred.predicted[depthIdx], pred.truth[depthIdx]);
    return (view === "predicted" ? pred.predicted : pred.truth)[depthIdx];
  }, [pred, view, depthIdx]);

  const scale = useMemo(() => {
    if (!pred) return { min: 0, max: 1, absMax: 1 };
    const { min, max } = gridExtent(pred.truth[depthIdx]);
    const err = errorGrid(pred.predicted[depthIdx], pred.truth[depthIdx]);
    const e = gridExtent(err);
    const absMax = Math.max(Math.abs(e.min), Math.abs(e.max), 0.1);
    return { min, max, absMax };
  }, [pred, depthIdx]);

  if (error) {
    return (
      <div className="mx-auto max-w-2xl p-8 text-slate-300">
        <p className="rounded-lg bg-rose-950/50 p-4 ring-1 ring-rose-500/30">
          Could not load model outputs: {error}. Run the ML pipeline
          (download, harmonize, train) so public/data is populated.
        </p>
      </div>
    );
  }
  if (!meta || !metrics || !pred) {
    return <div className="p-8 text-slate-400">Loading model outputs...</div>;
  }

  const profilePred = picked ? pred.predicted.map((d) => d[picked.row][picked.col]) : [];
  const profileTruth = picked ? pred.truth.map((d) => d[picked.row][picked.col]) : [];
  const pickedLat = picked ? pred.lat[picked.row] : null;
  const pickedLon = picked ? pred.lon[picked.col] : null;
  const activeDepth = meta.depths[depthIdx];

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="animate-fade-up mb-8 flex flex-wrap items-start justify-between gap-6">
        <div className="max-w-xl">
          <div className="mb-2 inline-flex items-center gap-1.5 rounded-full bg-cyan-500/10 px-2.5 py-1 text-[11px] font-medium tracking-wide text-cyan-300 ring-1 ring-cyan-500/20">
            North Indian Ocean &middot; {meta.region.resolution}&deg; daily
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-50 sm:text-3xl">
            Subsurface temperature reconstruction
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-slate-400">
            A satellite-embedding deep network turns surface fields — SST, SSH,
            SSS, currents — into a full temperature profile, validated against
            independent Argo floats.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
              source === "api"
                ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
                : "bg-slate-700/50 text-slate-300 ring-1 ring-white/10"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                source === "api" ? "bg-emerald-400 animate-pulse-slow" : "bg-slate-400"
              }`}
            />
            {source === "api" ? "Live inference" : "Static sample"}
          </span>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <span className="text-slate-500">Model</span>
            <select
              value={model}
              onChange={(e) => {
                const m = e.target.value as ModelName;
                onModel(m);
                if (m !== "cnn" && view === "embedding") setView("predicted");
                if (m !== "cnn" && validation === "argo") setValidation("holdout");
              }}
              disabled={models.length < 2}
              className="rounded-md bg-slate-800 px-2 py-1.5 text-slate-100 ring-1 ring-white/10 transition hover:ring-white/20 disabled:opacity-50"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {MODEL_LABEL[m]}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <span className="text-slate-500">Date</span>
            <select
              value={date}
              onChange={(e) => onDate(e.target.value)}
              disabled={source !== "api"}
              className="rounded-md bg-slate-800 px-2 py-1.5 text-slate-100 ring-1 ring-white/10 transition hover:ring-white/20 disabled:opacity-50"
            >
              {meta.times.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <div className="animate-fade-up mb-4 flex flex-wrap items-center gap-3" style={{ animationDelay: "60ms" }}>
        <span className="text-xs uppercase tracking-wide text-slate-500">Validation</span>
        <div className="inline-flex rounded-lg bg-slate-800 p-0.5 text-sm">
          {(["holdout", "argo"] as Validation[]).map((v) => {
            const enabled = v !== "argo" || model === "cnn";
            return (
              <button
                key={v}
                onClick={() => enabled && setValidation(v)}
                disabled={!enabled}
                className={`rounded-md px-3 py-1 transition disabled:opacity-40 ${
                  validation === v ? "bg-slate-600 text-white" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {v === "holdout" ? "GLORYS holdout" : "Independent ARGO"}
              </button>
            );
          })}
        </div>
        <span className="text-xs text-slate-400">
          {validation === "argo"
            ? "vs IPRC/APDRC gridded ARGO floats (independent of training)"
            : "vs held-out GLORYS days (last 7 of the month)"}
        </span>
      </div>

      <div
        className="animate-fade-up mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4"
        style={{ animationDelay: "100ms" }}
      >
        <Kpi label="Overall RMSE" value={`${metrics.overall.rmse.toFixed(3)}`} unit="degC" icon="rmse" />
        <Kpi label="Correlation" value={metrics.overall.corr.toFixed(3)} icon="corr" />
        <Kpi label="Bias" value={`${metrics.overall.bias >= 0 ? "+" : ""}${metrics.overall.bias.toFixed(3)}`} unit="degC" icon="bias" />
        <Kpi label="Depth levels" value={`${meta.depths.length}`} unit="0-1000m" icon="depth" />
      </div>

      <div
        className="animate-fade-up grid grid-cols-1 gap-6 lg:grid-cols-3"
        style={{ animationDelay: "140ms" }}
      >
        <section className="glass-panel rounded-2xl p-4 transition-shadow lg:col-span-2">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div className="inline-flex rounded-lg bg-slate-800 p-0.5 text-sm">
              {(["predicted", "truth", "error", "embedding"] as View[]).map((v) => {
                const enabled = v !== "embedding" || model === "cnn";
                return (
                  <button
                    key={v}
                    onClick={() => enabled && setView(v)}
                    disabled={!enabled}
                    className={`rounded-md px-3 py-1 capitalize transition disabled:opacity-40 ${
                      view === v ? "bg-slate-600 text-white" : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {v}
                  </button>
                );
              })}
            </div>
            <span className="inline-flex items-center gap-1.5 text-sm text-slate-400">
              {switching && (
                <span className="h-3 w-3 animate-spin rounded-full border-[1.5px] border-slate-600 border-t-amber-400" />
              )}
              {switching
                ? "Reconstructing..."
                : view === "embedding"
                  ? `${pred.date} . latent embedding`
                  : `${pred.date} . ${activeDepth} m depth`}
            </span>
          </div>

          <div className={`transition-opacity duration-300 ${switching ? "opacity-50" : "opacity-100"}`}>
            {view === "embedding" ? (
              <div>
                {embedding ? (
                  <EmbeddingCanvas rgb={embedding.rgb} />
                ) : (
                  <div className="flex aspect-241/101 items-center justify-center rounded-lg bg-slate-800/50 text-sm text-slate-400">
                    Computing embedding...
                  </div>
                )}
                <p className="mt-3 text-xs leading-5 text-slate-400">
                  The learned satellite embedding: the CNN bottleneck reduced to RGB
                  by PCA. Regions the model represents similarly share a colour,
                  exposing water masses and dynamic structures it uses to infer the
                  subsurface.
                </p>
              </div>
            ) : (
              <>
                <HeatmapCanvas
                  grid={displayGrid ?? pred.truth[depthIdx]}
                  mode={view === "error" ? "diverging" : "thermal"}
                  min={scale.min}
                  max={scale.max}
                  absMax={scale.absMax}
                  picked={picked}
                  onPick={(row, col) => onPick({ row, col })}
                />
                <div className="mt-3">
                  <Legend
                    mode={view === "error" ? "diverging" : "thermal"}
                    min={scale.min}
                    max={scale.max}
                    absMax={scale.absMax}
                  />
                </div>
              </>
            )}
          </div>

          <div className={`mt-4 ${view === "embedding" ? "hidden" : ""}`}>
            <label className="flex items-center justify-between text-sm text-slate-300">
              <span>Depth</span>
              <span className="font-mono text-amber-400">{activeDepth} m</span>
            </label>
            <input
              type="range"
              min={0}
              max={meta.depths.length - 1}
              value={depthIdx}
              onChange={(e) => setDepthIdx(Number(e.target.value))}
              className="mt-2 w-full accent-amber-500"
            />
          </div>
        </section>

        <section className="glass-panel rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 text-cyan-400" fill="none">
                <path d="M12 3v18M6 8l6-5 6 5M6 16l6 5 6-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Vertical profile
            </h2>
            {pickedLat !== null && (
              <span className="rounded-md bg-slate-800/80 px-1.5 py-0.5 font-mono text-[11px] text-slate-400">
                {pickedLat.toFixed(2)} N, {pickedLon?.toFixed(2)} E
              </span>
            )}
          </div>
          <ProfileChart depths={meta.depths} predicted={profilePred} truth={profileTruth} />
          <div className="mt-2 flex gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 bg-sky-400" /> Truth (GLORYS)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 bg-amber-500" /> Predicted
            </span>
          </div>
        </section>
      </div>

      <section
        className="glass-panel animate-fade-up mt-6 rounded-2xl p-4"
        style={{ animationDelay: "180ms" }}
      >
        <h2 className="mb-1 flex items-center gap-1.5 text-sm font-medium text-slate-200">
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 text-emerald-400" fill="none">
            <path d="M4 20V10M10 20V4M16 20v-7M22 20v-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Skill by depth
        </h2>
        <p className="mb-3 text-xs text-slate-400">
          Bars: RMSE per depth. Line: correlation (right axis). The active depth is highlighted.
        </p>
        <SkillChart perDepth={metrics.per_depth} activeDepth={activeDepth} />
      </section>

      <footer className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-white/5 pt-4 text-xs text-slate-500">
        <span>SIH 2026 &middot; Problem 26066 &middot; {MODEL_LABEL[model]} model</span>
        <span>Validation is a temporal holdout of the last 7 days.</span>
      </footer>
    </div>
  );
}

const KPI_ICON: Record<string, ReactNode> = {
  rmse: (
    <path d="M3 12h4l2-7 4 14 2-7h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  ),
  corr: (
    <path d="M4 20 20 4M4 4l4 4M20 20l-4-4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  ),
  bias: (
    <path d="M12 3v18M5 10l7-7 7 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  ),
  depth: (
    <path d="M3 15c2 1.5 4 1.5 6 0s4-1.5 6 0 4 1.5 6 0M12 3v9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  ),
};

function Kpi({ label, value, unit, icon }: { label: string; value: string; unit?: string; icon?: string }) {
  return (
    <div className="glass-panel group rounded-2xl p-4 transition-transform duration-200 hover:-translate-y-0.5">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
        {icon && (
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 text-slate-600 transition-colors group-hover:text-cyan-400" fill="none">
            {KPI_ICON[icon]}
          </svg>
        )}
      </div>
      <div className="mt-1 flex items-baseline gap-1">
        <span className="text-xl font-semibold text-slate-50 sm:text-2xl">{value}</span>
        {unit && <span className="text-xs text-slate-500">{unit}</span>}
      </div>
    </div>
  );
}
