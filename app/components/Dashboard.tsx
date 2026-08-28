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
import Column3D from "./Column3D";
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
  const [profile3d, setProfile3d] = useState(true);

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
        <p className="max-w-md text-sm text-[color:var(--muted)]">
          Subsurface temperature reconstructed from surface satellite fields.
          North Indian Ocean, {meta.region.resolution} deg daily.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
              source === "api"
                ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
                : "bg-white/5 text-[color:var(--muted)] ring-1 ring-[color:var(--line)]"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                source === "api" ? "bg-emerald-400 shadow-[0_0_8px_2px_rgba(52,211,153,0.6)]" : "bg-slate-400"
              }`}
            />
            {source === "api" ? "Live inference" : "Static sample"}
          </span>
          <label className="flex items-center gap-2 text-sm">
            <span className="microlabel">Model</span>
            <select
              value={model}
              onChange={(e) => {
                const m = e.target.value as ModelName;
                setModel(m);
                if (m !== "cnn" && view === "embedding") setView("predicted");
                if (m !== "cnn" && validation === "argo") setValidation("holdout");
              }}
              disabled={models.length < 2}
              className="disabled:opacity-50"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {MODEL_LABEL[m]}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <span className="microlabel">Date</span>
            <select
              value={date}
              onChange={(e) => setDate(e.target.value)}
              disabled={source !== "api"}
              className="disabled:opacity-50"
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

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <span className="microlabel">Validation</span>
        <div className="segmented">
          {(["holdout", "argo"] as Validation[]).map((v) => {
            const enabled = v !== "argo" || model === "cnn";
            return (
              <button
                key={v}
                onClick={() => enabled && setValidation(v)}
                disabled={!enabled}
                data-active={validation === v}
                className="pill"
              >
                {v === "holdout" ? "GLORYS holdout" : "Independent ARGO"}
              </button>
            );
          })}
        </div>
        <span className="text-xs text-[color:var(--muted)]">
          {validation === "argo"
            ? "vs IPRC/APDRC gridded ARGO floats (independent of training)"
            : "vs held-out GLORYS days (last 7 of the month)"}
        </span>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi label="Overall RMSE" value={metrics.overall.rmse.toFixed(3)} unit="degC" tone="warm" />
        <Kpi label="Correlation" value={metrics.overall.corr.toFixed(3)} tone="accent" />
        <Kpi label="Bias" value={metrics.overall.bias.toFixed(3)} unit="degC" />
        <Kpi label="Depth levels" value={`${meta.depths.length}`} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <section className="card p-4 lg:col-span-2">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div className="segmented">
              {(["predicted", "truth", "error", "embedding"] as View[]).map((v) => {
                const enabled = v !== "embedding" || model === "cnn";
                return (
                  <button
                    key={v}
                    onClick={() => enabled && setView(v)}
                    disabled={!enabled}
                    data-active={view === v}
                    className="pill capitalize"
                  >
                    {v}
                  </button>
                );
              })}
            </div>
            <span className="text-sm text-[color:var(--muted)]">
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
                <div className="flex aspect-[241/101] items-center justify-center rounded-lg bg-white/5 text-sm text-[color:var(--muted)]">
                  Computing embedding...
                </div>
              )}
              <p className="mt-3 text-xs leading-5 text-[color:var(--muted)]">
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
            <label className="flex items-center justify-between text-sm text-[color:var(--muted)]">
              <span>Depth</span>
              <span className="metric font-mono text-[color:var(--accent-warm)]">{activeDepth} m</span>
            </label>
            <input
              type="range"
              min={0}
              max={meta.depths.length - 1}
              value={depthIdx}
              onChange={(e) => setDepthIdx(Number(e.target.value))}
              className="mt-2 w-full"
            />
          </div>
        </section>

        <section className="card p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-medium text-[color:var(--text)]">Vertical profile</h2>
            <div className="segmented">
              <button className="pill" data-active={profile3d} onClick={() => setProfile3d(true)}>
                3D
              </button>
              <button className="pill" data-active={!profile3d} onClick={() => setProfile3d(false)}>
                2D
              </button>
            </div>
          </div>
          {pickedLat !== null && (
            <div className="metric mb-1 font-mono text-xs text-[color:var(--faint)]">
              {pickedLat.toFixed(2)} N, {pickedLon?.toFixed(2)} E
            </div>
          )}
          {profile3d ? (
            <Column3D
              depths={meta.depths}
              min={scale.min}
              max={scale.max}
              columns={[
                { label: "Predicted", temps: profilePred },
                { label: "Truth", temps: profileTruth },
              ]}
            />
          ) : (
            <>
              <ProfileChart depths={meta.depths} predicted={profilePred} truth={profileTruth} />
              <div className="mt-2 flex gap-4 text-xs text-[color:var(--muted)]">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-4 bg-sky-400" /> Truth (GLORYS)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-4 bg-amber-500" /> Predicted
                </span>
              </div>
            </>
          )}
        </section>
      </div>

      <section className="card mt-6 p-4">
        <h2 className="mb-1 text-sm font-medium text-[color:var(--text)]">Skill by depth</h2>
        <p className="mb-3 text-xs text-[color:var(--muted)]">
          Bars: RMSE per depth. Line: correlation (right axis). The active depth is highlighted.
        </p>
        <SkillChart perDepth={metrics.per_depth} activeDepth={activeDepth} />
      </section>

      <footer className="mt-8 text-xs text-[color:var(--faint)]">
        SIH 2026 . Problem 26066 . {MODEL_LABEL[model]} model.
      </footer>
    </div>
  );
}

function Kpi({
  label,
  value,
  unit,
  tone,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "warm" | "accent";
}) {
  const color =
    tone === "warm" ? "text-[color:var(--accent-warm)]" : tone === "accent" ? "text-sky-300" : "text-[color:var(--text)]";
  return (
    <div className="card p-4">
      <div className="microlabel">{label}</div>
      <div className={`metric mt-1 text-2xl font-semibold ${color}`}>
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-[color:var(--faint)]">{unit}</span>}
      </div>
    </div>
  );
}
