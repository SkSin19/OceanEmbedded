"use client";

import { estimateThermocline } from "../lib/depth";
import type { ColumnFrame } from "../lib/api";

interface Props {
  depths: number[];
  a: ColumnFrame | null;
  b: ColumnFrame | null;
  dates: string[];
  aIndex: number;
  onAIndex: (i: number) => void;
  ready: boolean[];
}

function surface(p: (number | null)[]): number | null {
  return p.find((v) => v !== null) ?? null;
}

function meanDelta(x: (number | null)[], y: (number | null)[]): number | null {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const a = x[i];
    const b = y[i];
    if (a === null || b === null) continue;
    sum += b - a;
    n++;
  }
  return n ? sum / n : null;
}

function peakDelta(
  x: (number | null)[],
  y: (number | null)[],
  depths: number[]
): { value: number; depth: number } | null {
  let best: { value: number; depth: number } | null = null;
  for (let i = 0; i < Math.min(x.length, y.length, depths.length); i++) {
    const a = x[i];
    const b = y[i];
    if (a === null || b === null) continue;
    const d = b - a;
    if (!best || Math.abs(d) > Math.abs(best.value)) best = { value: d, depth: depths[i] };
  }
  return best;
}

export default function DeltaCards({
  depths,
  a,
  b,
  dates,
  aIndex,
  onAIndex,
  ready,
}: Props) {
  if (!a || !b) {
    return (
      <p className="text-[11px] leading-relaxed text-slate-500">
        Waiting for both frames to load.
      </p>
    );
  }

  const sA = surface(a.predicted);
  const sB = surface(b.predicted);
  const tA = estimateThermocline(a.predicted, depths);
  const tB = estimateThermocline(b.predicted, depths);
  const mean = meanDelta(a.predicted, b.predicted);
  const peak = peakDelta(a.predicted, b.predicted, depths);

  const surfD = sA !== null && sB !== null ? sB - sA : null;
  const thermoD = tA && tB ? tB.depth - tA.depth : null;
  const gradD = tA && tB ? tB.gradient - tA.gradient : null;

  return (
    <div className="space-y-3">
      <label className="block">
        <span className="mb-1 block text-[11px] text-slate-400">
          Baseline frame{" "}
          <span className="font-mono text-slate-300">{dates[aIndex] ?? "--"}</span>
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, dates.length - 1)}
          value={aIndex}
          onChange={(e) => onAIndex(Number(e.target.value))}
          className="w-full accent-sky-400"
        />
        {!ready[aIndex] && (
          <span className="text-[10px] text-amber-400">frame not loaded yet</span>
        )}
      </label>

      <div className="grid grid-cols-2 gap-2">
        <Card
          label="Surface dT"
          value={surfD === null ? "--" : `${surfD >= 0 ? "+" : ""}${surfD.toFixed(2)}`}
          unit="degC"
          tone={surfD === null ? "flat" : surfD > 0 ? "warm" : "cool"}
        />
        <Card
          label="Thermocline"
          value={thermoD === null ? "--" : `${thermoD >= 0 ? "+" : ""}${thermoD}`}
          unit="m shift"
          tone={thermoD === null || thermoD === 0 ? "flat" : thermoD > 0 ? "cool" : "warm"}
        />
        <Card
          label="Column mean dT"
          value={mean === null ? "--" : `${mean >= 0 ? "+" : ""}${mean.toFixed(2)}`}
          unit="degC"
          tone={mean === null ? "flat" : mean > 0 ? "warm" : "cool"}
        />
        <Card
          label="Peak dT"
          value={peak === null ? "--" : `${peak.value >= 0 ? "+" : ""}${peak.value.toFixed(2)}`}
          unit={peak === null ? "degC" : `degC @ ${peak.depth} m`}
          tone={peak === null ? "flat" : peak.value > 0 ? "warm" : "cool"}
        />
      </div>

      <div className="rounded-lg border border-white/10 bg-slate-900/40 px-2.5 py-2 font-mono text-[10.5px] leading-relaxed text-slate-400">
        <div>
          {a.date} &rarr; {b.date}
        </div>
        <div>
          gradient {tA ? tA.gradient.toFixed(3) : "--"} &rarr;{" "}
          {tB ? tB.gradient.toFixed(3) : "--"} degC/m
          {gradD !== null && (
            <span className={gradD < 0 ? " text-amber-400" : " text-sky-400"}>
              {" "}
              ({gradD >= 0 ? "+" : ""}
              {gradD.toFixed(3)})
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function Card({
  label,
  value,
  unit,
  tone,
}: {
  label: string;
  value: string;
  unit: string;
  tone: "warm" | "cool" | "flat";
}) {
  const ring =
    tone === "warm"
      ? "ring-amber-500/25 bg-amber-500/[0.07]"
      : tone === "cool"
        ? "ring-cyan-500/25 bg-cyan-500/[0.07]"
        : "ring-white/10 bg-white/[0.03]";
  const text = tone === "warm" ? "text-amber-300" : tone === "cool" ? "text-cyan-300" : "text-slate-300";
  return (
    <div className={`rounded-xl px-2.5 py-2 ring-1 backdrop-blur-md ${ring}`}>
      <div className="text-[9.5px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`font-mono text-base font-semibold leading-tight ${text}`}>{value}</div>
      <div className="text-[9.5px] text-slate-500">{unit}</div>
    </div>
  );
}
