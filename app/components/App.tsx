"use client";

import { useEffect, useState } from "react";
import type { Meta, ModelName } from "../lib/types";
import { detectSource, loadMeta, loadModels, type Source } from "../lib/api";
import Dashboard from "./Dashboard";
import Playground from "./Playground";

type Tab = "explorer" | "playground";

export default function App() {
  const [source, setSource] = useState<Source | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [models, setModels] = useState<ModelName[]>(["baseline"]);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("explorer");

  useEffect(() => {
    (async () => {
      try {
        const src = await detectSource();
        const [m, avail] = await Promise.all([loadMeta(src), loadModels(src)]);
        setSource(src);
        setMeta(m);
        setModels(avail);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-200">
      <nav className="sticky top-0 z-10 border-b border-white/10 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold tracking-tight text-slate-50">OceanEmbed</span>
            <span className="hidden text-xs text-slate-500 sm:inline">
              subsurface temperature from satellite surface fields
            </span>
          </div>
          <div className="inline-flex rounded-lg bg-slate-800 p-0.5 text-sm">
            {(["explorer", "playground"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-md px-3 py-1 capitalize transition ${
                  tab === t ? "bg-slate-600 text-white" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {error ? (
        <div className="mx-auto max-w-2xl p-8 text-slate-300">
          <p className="rounded-lg bg-rose-950/50 p-4 ring-1 ring-rose-500/30">
            Could not load model outputs: {error}. Run the ML pipeline so
            public/data is populated, or start the API.
          </p>
        </div>
      ) : source === null || meta === null ? (
        <div className="p-8 text-slate-400">Loading model outputs...</div>
      ) : tab === "explorer" ? (
        <Dashboard source={source} meta={meta} models={models} />
      ) : (
        <Playground source={source} meta={meta} models={models} />
      )}
    </div>
  );
}
