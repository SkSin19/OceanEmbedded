"use client";

import { useEffect, useMemo, useState } from "react";
import type { Embedding, Grid, Meta, Metrics, ModelName, Prediction } from "../lib/types";
import {
  loadEmbedding,
  loadMetrics,
  loadPrediction,
  type Source,
  type Validation,
} from "../lib/api";
import { errorGrid, gridExtent, nearestOcean } from "../lib/grid";
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
}

export default function Dashboard({ source, meta, models }: DashboardProps) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [depthIdx, setDepthIdx] = useState(0);
  const [view, setView] = useState<View>("predicted");
  const [picked, setPicked] = useState<{ row: number; col: number } | null>(null);

  const [date, setDate] = useState<string>(meta.times[meta.times.length - 1]);
  const [switching, setSwitching] = useState(false);
  const [model, setModel] = useState<ModelName>(models.includes("cnn") ? "cnn" : "baseline");
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

  // Reconstruction follows date + model.
  useEffect(() => {
    if (!date) return;
    let active = true;
    (async () => {
      setSwitching(true);
      try {
        const p = await loadPrediction(source, date, model);
        if (!active) return;
        setPred(p);
        setPicked((prev) => prev ?? nearestOcean(p.truth[0], p.lat, p.lon, 15, 65));
      } catch (e) {
        if (active) setError(String(e));
      } finally {
        if (active) setSwitching(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [source, date, model]);

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
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <p className="max-w-md text-sm text-slate-400">
          Subsurface temperature reconstructed from surface satellite fields.
          North Indian Ocean, {meta.region.resolution} deg daily.
        </p>
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
                source === "api" ? "bg-emerald-400" : "bg-slate-400"
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
                setModel(m);
                if (m !== "cnn" && view === "embedding") setView("predicted");
                if (m !== "cnn" && validation === "argo") setValidation("holdout");
              }}
              disabled={models.length < 2}
              className="rounded-md bg-slate-800 px-2 py-1 text-slate-100 ring-1 ring-white/10 disabled:opacity-50"
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
              onChange={(e) => setDate(e.target.value)}
              disabled={source !== "api"}
              className="rounded-md bg-slate-800 px-2 py-1 text-slate-100 ring-1 ring-white/10 disabled:opacity-50"
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

      <div className="mb-3 flex flex-wrap items-center gap-3">
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

      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi label="Overall RMSE" value={`${metrics.overall.rmse.toFixed(3)} degC`} />
        <Kpi label="Correlation" value={metrics.overall.corr.toFixed(3)} />
        <Kpi label="Bias" value={`${metrics.overall.bias.toFixed(3)} degC`} />
        <Kpi label="Depth levels" value={`${meta.depths.length}`} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <section className="lg:col-span-2 rounded-xl bg-slate-900/60 p-4 ring-1 ring-white/10">
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
            <span className="text-sm text-slate-400">
              {switching
                ? "Reconstructing..."
                : view === "embedding"
                  ? `${pred.date} . latent embedding`
                  : `${pred.date} . ${activeDepth} m depth`}
            </span>
          </div>

          {view === "embedding" ? (
            <div>
              {embedding ? (
                <EmbeddingCanvas rgb={embedding.rgb} />
              ) : (
                <div className="flex aspect-[241/101] items-center justify-center rounded-lg bg-slate-800/50 text-sm text-slate-400">
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
                onPick={(row, col) => setPicked({ row, col })}
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

        <section className="rounded-xl bg-slate-900/60 p-4 ring-1 ring-white/10">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-slate-200">Vertical profile</h2>
            {pickedLat !== null && (
              <span className="font-mono text-xs text-slate-400">
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

      <section className="mt-6 rounded-xl bg-slate-900/60 p-4 ring-1 ring-white/10">
        <h2 className="mb-1 text-sm font-medium text-slate-200">Skill by depth</h2>
        <p className="mb-3 text-xs text-slate-400">
          Bars: RMSE per depth. Line: correlation (right axis). The active depth is highlighted.
        </p>
        <SkillChart perDepth={metrics.per_depth} activeDepth={activeDepth} />
      </section>

      <footer className="mt-8 text-xs text-slate-500">
        SIH 2026 . Problem 26066 . {MODEL_LABEL[model]} model. Validation is a
        temporal holdout of the last 7 days.
      </footer>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-slate-900/60 p-4 ring-1 ring-white/10">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-lg font-semibold text-slate-50">{value}</div>
    </div>
  );
}
