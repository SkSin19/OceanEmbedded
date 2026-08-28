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
    <div className="min-h-screen font-sans text-[color:var(--text)]">
      <nav className="sticky top-0 z-20 border-b border-[color:var(--line)] bg-[color:var(--bg)]/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-sky-400/30 to-indigo-500/20 ring-1 ring-sky-400/30">
              <span className="h-2.5 w-2.5 rounded-full bg-gradient-to-br from-sky-300 to-cyan-500 shadow-[0_0_12px_2px_rgba(56,189,248,0.6)]" />
            </span>
            <div className="leading-tight">
              <div className="bg-gradient-to-r from-sky-200 to-indigo-200 bg-clip-text text-lg font-semibold tracking-tight text-transparent">
                OceanEmbed
              </div>
              <div className="hidden text-[11px] text-[color:var(--faint)] sm:block">
                subsurface temperature from satellite surface fields
              </div>
            </div>
          </div>
          <div className="segmented">
            {(["explorer", "playground"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                data-active={tab === t}
                className="pill capitalize"
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {error ? (
        <div className="mx-auto max-w-2xl p-8">
          <p className="card p-4 text-rose-200">
            Could not load model outputs: {error}. Run the ML pipeline so
            public/data is populated, or start the API.
          </p>
        </div>
      ) : source === null || meta === null ? (
        <div className="flex min-h-[60vh] items-center justify-center gap-3 text-[color:var(--muted)]">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-sky-400/40 border-t-sky-300" />
          Loading model outputs...
        </div>
      ) : tab === "explorer" ? (
        <Dashboard source={source} meta={meta} models={models} />
      ) : (
        <Playground source={source} meta={meta} models={models} />
      )}
    </div>
  );
}
