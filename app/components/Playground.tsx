"use client";

import { useEffect, useRef, useState } from "react";
import type { Meta, ModelName, Reconstruction } from "../lib/types";
import { reconstructImage, reconstructWhatIf, type Source } from "../lib/api";
import { gridExtent, nearestOcean } from "../lib/grid";
import { MODEL_LABEL } from "../lib/labels";
import HeatmapCanvas from "./HeatmapCanvas";
import ProfileChart from "./ProfileChart";
import Column3D from "./Column3D";
import Legend from "./Legend";

type Mode = "whatif" | "image";

interface Props {
  source: Source;
  meta: Meta;
  models: ModelName[];
}

export default function Playground({ source, meta, models }: Props) {
  const [model, setModel] = useState<ModelName>(models.includes("cnn") ? "cnn" : "baseline");
  const [date, setDate] = useState(meta.times[meta.times.length - 1]);
  const [mode, setMode] = useState<Mode>("whatif");
  const [sstOffset, setSstOffset] = useState(0);
  const [sshOffset, setSshOffset] = useState(0);
  const [result, setResult] = useState<Reconstruction | null>(null);
  const [depthIdx, setDepthIdx] = useState(0);
  const [picked, setPicked] = useState<{ row: number; col: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [imageName, setImageName] = useState<string | null>(null);
  const [isColormap, setIsColormap] = useState(false);
  const [profile3d, setProfile3d] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // What-if runs automatically (debounced). Image runs on upload only.
  useEffect(() => {
    if (source !== "api" || mode !== "whatif") return;
    let active = true;
    const id = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await reconstructWhatIf(date, model, sstOffset, sshOffset);
        if (!active) return;
        setResult(r);
        setPicked((p) => p ?? nearestOcean(r.baseline[0], r.lat, r.lon, 15, 65));
      } catch (e) {
        if (active) setError(String(e));
      } finally {
        if (active) setLoading(false);
      }
    }, 250);
    return () => {
      active = false;
      clearTimeout(id);
    };
  }, [source, mode, date, model, sstOffset, sshOffset]);

  async function onImage(file: File, colormap: "grayscale" | "thermal") {
    setImageName(file.name);
    setLoading(true);
    setError(null);
    try {
      const r = await reconstructImage(date, model, file, colormap);
      setResult(r);
      setPicked((p) => p ?? nearestOcean(r.baseline[0], r.lat, r.lon, 15, 65));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  if (source !== "api") {
    return (
      <div className="mx-auto max-w-2xl p-8">
        <p className="card p-4 text-sm text-amber-200">
          The playground runs the model on modified inputs, so it needs the live
          backend. Start it with{" "}
          <code className="rounded bg-black/40 px-1">uvicorn serve.app:app --port 8010</code>{" "}
          and reload.
        </p>
      </div>
    );
  }

  const depths = meta.depths;
  const activeDepth = depths[depthIdx];
  const predScale = result ? gridExtent(result.baseline[depthIdx]) : { min: 0, max: 1 };
  const sstScale = result ? gridExtent(result.sst_input) : { min: 0, max: 1 };

  const modProfile =
    result && picked ? result.predicted.map((d) => d[picked.row][picked.col]) : [];
  const baseProfile =
    result && picked ? result.baseline.map((d) => d[picked.row][picked.col]) : [];
  const surfaceDelta =
    modProfile[0] !== null && baseProfile[0] !== null && modProfile.length > 0
      ? (modProfile[0] as number) - (baseProfile[0] as number)
      : null;

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6">
        <p className="max-w-2xl text-sm text-[color:var(--muted)]">
          Feed the model a modified surface state and watch the subsurface it
          predicts. Shift SST to simulate a marine heatwave, or upload an image as
          the SST field. The two columns are the new reconstruction and the
          unmodified day.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <section className="card p-4">
          <h2 className="mb-3 text-sm font-medium text-[color:var(--text)]">Controls</h2>

          <div className="segmented mb-4">
            {(["whatif", "image"] as Mode[]).map((m) => (
              <button key={m} onClick={() => setMode(m)} data-active={mode === m} className="pill">
                {m === "whatif" ? "Heatwave what-if" : "Upload image"}
              </button>
            ))}
          </div>

          <div className="space-y-3 text-sm">
            <label className="flex items-center justify-between gap-2">
              <span className="microlabel">Base day</span>
              <select value={date} onChange={(e) => setDate(e.target.value)}>
                {meta.times.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center justify-between gap-2">
              <span className="microlabel">Model</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value as ModelName)}
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
          </div>

          {mode === "whatif" ? (
            <div className="mt-4 space-y-4">
              <div>
                <label className="flex items-center justify-between text-sm text-slate-300">
                  <span>SST shift</span>
                  <span className="font-mono text-amber-400">
                    {sstOffset > 0 ? "+" : ""}
                    {sstOffset.toFixed(1)} degC
                  </span>
                </label>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.5}
                  value={sstOffset}
                  onChange={(e) => setSstOffset(Number(e.target.value))}
                  className="mt-2 w-full accent-amber-500"
                />
              </div>
              <div>
                <label className="flex items-center justify-between text-sm text-slate-300">
                  <span>SSH shift</span>
                  <span className="font-mono text-amber-400">
                    {sshOffset > 0 ? "+" : ""}
                    {sshOffset.toFixed(2)} m
                  </span>
                </label>
                <input
                  type="range"
                  min={-0.3}
                  max={0.3}
                  step={0.05}
                  value={sshOffset}
                  onChange={(e) => setSshOffset(Number(e.target.value))}
                  className="mt-2 w-full accent-amber-500"
                />
              </div>
              <button
                onClick={() => {
                  setSstOffset(0);
                  setSshOffset(0);
                }}
                className="rounded-md bg-slate-800 px-3 py-1 text-sm text-slate-300 ring-1 ring-white/10 hover:text-white"
              >
                Reset
              </button>
            </div>
          ) : (
            <div className="mt-4">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onImage(f, isColormap ? "thermal" : "grayscale");
                }}
                className="hidden"
              />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full rounded-lg border border-dashed border-white/20 bg-slate-800/50 px-4 py-6 text-sm text-slate-300 hover:border-amber-500/50 hover:text-white"
              >
                {imageName ? `Selected: ${imageName}` : "Click to upload an image"}
              </button>
              <label className="mt-3 flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={isColormap}
                  onChange={(e) => setIsColormap(e.target.checked)}
                  className="accent-amber-500"
                />
                Image uses the SST colormap (decode colours back to temperature)
              </label>
              <p className="mt-2 text-xs text-slate-500">
                Off: brightness is read as temperature (grayscale SST). On: a colour
                SST map is decoded via the dashboard colour scale, so a real-looking
                colour map reconstructs correctly. The field is fed to the model over
                the ocean mask of the chosen day.
              </p>
            </div>
          )}

          {loading && <p className="mt-3 text-xs text-amber-400">Reconstructing...</p>}
          {error && <p className="mt-3 text-xs text-rose-400">{error}</p>}
          {surfaceDelta !== null && (
            <p className="mt-4 text-xs text-slate-400">
              At the selected point, the surface changed by{" "}
              <span className="font-mono text-amber-400">
                {surfaceDelta > 0 ? "+" : ""}
                {surfaceDelta.toFixed(2)} degC
              </span>
              . Read the profile to see how deep the change propagates.
            </p>
          )}
        </section>

        <section className="lg:col-span-2 card p-4">
          {!result ? (
            <div className="flex h-80 items-center justify-center text-sm text-slate-400">
              Adjust a control to run the model.
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                  {mode === "image" ? "SST input (from image)" : "SST input"}
                </h3>
                <HeatmapCanvas
                  grid={result.sst_input}
                  mode="thermal"
                  min={sstScale.min}
                  max={sstScale.max}
                  absMax={1}
                  picked={picked}
                  onPick={(row, col) => setPicked({ row, col })}
                />
                <div className="mt-2">
                  <Legend mode="thermal" min={sstScale.min} max={sstScale.max} absMax={1} />
                </div>
              </div>
              <div>
                <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                  Reconstruction at {activeDepth} m
                </h3>
                <HeatmapCanvas
                  grid={result.predicted[depthIdx]}
                  mode="thermal"
                  min={predScale.min}
                  max={predScale.max}
                  absMax={1}
                  picked={picked}
                  onPick={(row, col) => setPicked({ row, col })}
                />
                <div className="mt-3">
                  <input
                    type="range"
                    min={0}
                    max={depths.length - 1}
                    value={depthIdx}
                    onChange={(e) => setDepthIdx(Number(e.target.value))}
                    className="w-full accent-amber-500"
                    aria-label="depth"
                  />
                </div>
              </div>
              <div className="md:col-span-2">
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="microlabel">Profile at the selected point</h3>
                  <div className="segmented">
                    <button className="pill" data-active={profile3d} onClick={() => setProfile3d(true)}>
                      3D
                    </button>
                    <button className="pill" data-active={!profile3d} onClick={() => setProfile3d(false)}>
                      2D
                    </button>
                  </div>
                </div>
                {profile3d ? (
                  <Column3D
                    depths={depths}
                    min={predScale.min}
                    max={predScale.max}
                    columns={[
                      { label: "Modified", temps: modProfile },
                      { label: "Unmodified", temps: baseProfile },
                    ]}
                  />
                ) : (
                  <>
                    <div className="mx-auto max-w-sm">
                      <ProfileChart depths={depths} predicted={modProfile} truth={baseProfile} />
                    </div>
                    <div className="mt-2 flex justify-center gap-4 text-xs text-[color:var(--muted)]">
                      <span className="flex items-center gap-1.5">
                        <span className="inline-block h-0.5 w-4 bg-sky-400" /> Unmodified day
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="inline-block h-0.5 w-4 bg-amber-500" /> Modified input
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
