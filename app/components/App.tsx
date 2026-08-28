"use client";

import { useEffect, useState } from "react";
import type { Meta, ModelName, Prediction } from "../lib/types";
import { detectSource, loadMeta, loadModels, loadPrediction, type Source } from "../lib/api";
import { nearestOcean } from "../lib/grid";
import Dashboard from "./Dashboard";
import Playground from "./Playground";
import OceanMap from "./OceanMap";
import VolumePage from "./VolumePage";

type Tab = "explorer" | "map" | "volume" | "playground";

const TABS: { id: Tab; label: string }[] = [
  { id: "explorer", label: "Explorer" },
  { id: "map", label: "Map" },
  { id: "volume", label: "3D" },
  { id: "playground", label: "Playground" },
];

export default function App() {
  const [source, setSource] = useState<Source | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [models, setModels] = useState<ModelName[]>(["baseline"]);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("explorer");

  // Shared reconstruction state, so switching tabs keeps the same day, model
  // and probe point rather than re-fetching and re-picking per view.
  const [date, setDate] = useState<string | null>(null);
  const [model, setModel] = useState<ModelName>("baseline");
  const [pred, setPred] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [picked, setPicked] = useState<{ row: number; col: number } | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const src = await detectSource();
        const [m, avail] = await Promise.all([loadMeta(src), loadModels(src)]);
        setSource(src);
        setMeta(m);
        setModels(avail);
        setModel(avail.includes("cnn") ? "cnn" : "baseline");
        setDate(m.times[m.times.length - 1]);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  // One reconstruction fetch feeds every tab.
  useEffect(() => {
    if (source === null || !date) return;
    let active = true;
    (async () => {
      setLoading(true);
      try {
        const p = await loadPrediction(source, date, model);
        if (!active) return;
        setPred(p);
        setPicked((prev) => prev ?? nearestOcean(p.truth[0], p.lat, p.lon, 15, 88));
      } catch (e) {
        if (active) setError(String(e));
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [source, date, model]);

  return (
    <div className="min-h-screen font-sans text-slate-200">
      <div className="bg-grid pointer-events-none fixed inset-0 z-0 h-[520px]" />

      <nav className="sticky top-0 z-20 border-b border-white/10 bg-slate-950/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <span className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-400 to-sky-600 shadow-[0_0_20px_-4px_rgba(34,211,238,0.7)]">
              <svg viewBox="0 0 24 24" className="h-4.5 w-4.5 text-slate-950" fill="none">
                <path
                  d="M3 15c2 1.5 4 1.5 6 0s4-1.5 6 0 4 1.5 6 0M3 19c2 1.5 4 1.5 6 0s4-1.5 6 0 4 1.5 6 0M12 3v8m0 0-3-3m3 3 3-3"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            <div className="flex flex-col leading-none">
              <span className="bg-gradient-to-r from-slate-50 to-slate-300 bg-clip-text text-lg font-semibold tracking-tight text-transparent">
                OceanEmbed
              </span>
              <span className="hidden text-[11px] text-slate-500 sm:inline">
                subsurface temperature from satellite surface fields
              </span>
            </div>
          </div>
          <div className="inline-flex rounded-lg bg-white/5 p-0.5 text-sm ring-1 ring-white/10">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`rounded-md px-3 py-1.5 transition-all duration-200 ${
                  tab === t.id
                    ? "bg-gradient-to-b from-slate-600 to-slate-700 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <div className="relative z-10">
        {error ? (
          <div className="mx-auto max-w-2xl p-8 text-slate-300">
            <p className="rounded-lg bg-rose-950/50 p-4 ring-1 ring-rose-500/30">
              Could not load model outputs: {error}. Run the ML pipeline so
              public/data is populated, or start the API.
            </p>
          </div>
        ) : source === null || meta === null || date === null ? (
          <div className="flex h-[70vh] items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-slate-400">
              <span className="h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />
              <span className="text-sm">Loading model outputs...</span>
            </div>
          </div>
        ) : tab === "explorer" ? (
          <Dashboard
            source={source}
            meta={meta}
            models={models}
            pred={pred}
            date={date}
            model={model}
            loading={loading}
            picked={picked}
            onDate={setDate}
            onModel={setModel}
            onPick={setPicked}
          />
        ) : tab === "map" ? (
          <OceanMap
            source={source}
            meta={meta}
            models={models}
            pred={pred}
            date={date}
            model={model}
            loading={loading}
            picked={picked}
            onPick={setPicked}
          />
        ) : tab === "volume" ? (
          <VolumePage
            source={source}
            meta={meta}
            pred={pred}
            date={date}
            model={model}
            loading={loading}
            picked={picked}
            onPick={setPicked}
            onDate={setDate}
          />
        ) : (
          <Playground source={source} meta={meta} models={models} />
        )}
      </div>
    </div>
  );
}
