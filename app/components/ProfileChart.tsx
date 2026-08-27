"use client";

interface Props {
  depths: number[];
  predicted: (number | null)[];
  truth: (number | null)[];
}

const W = 320;
const Hgt = 380;
const PAD = { top: 16, right: 16, bottom: 34, left: 44 };

export default function ProfileChart({ depths, predicted, truth }: Props) {
  const vals = [...predicted, ...truth].filter((v): v is number => v !== null);
  if (vals.length === 0) {
    return (
      <div className="flex h-[380px] items-center justify-center text-sm text-slate-400">
        Click a point on the map to see its profile.
      </div>
    );
  }
  const min = Math.floor(Math.min(...vals) - 1);
  const max = Math.ceil(Math.max(...vals) + 1);

  const plotW = W - PAD.left - PAD.right;
  const plotH = Hgt - PAD.top - PAD.bottom;
  const x = (t: number) => PAD.left + ((t - min) / (max - min)) * plotW;
  // Even spacing by depth index so shallow levels stay legible.
  const y = (i: number) => PAD.top + (i / (depths.length - 1)) * plotH;

  const path = (series: (number | null)[]) =>
    series
      .map((v, i) => (v === null ? null : `${x(v).toFixed(1)},${y(i).toFixed(1)}`))
      .filter(Boolean)
      .map((p, i) => `${i === 0 ? "M" : "L"}${p}`)
      .join(" ");

  const xTicks = 4;
  return (
    <svg viewBox={`0 0 ${W} ${Hgt}`} className="w-full" role="img" aria-label="Temperature profile">
      {Array.from({ length: xTicks + 1 }).map((_, i) => {
        const t = min + ((max - min) * i) / xTicks;
        return (
          <g key={i}>
            <line x1={x(t)} y1={PAD.top} x2={x(t)} y2={Hgt - PAD.bottom} stroke="#1e293b" />
            <text x={x(t)} y={Hgt - PAD.bottom + 16} fill="#64748b" fontSize="10" textAnchor="middle">
              {t.toFixed(0)}
            </text>
          </g>
        );
      })}
      {depths.map((d, i) => (
        <text key={d} x={PAD.left - 6} y={y(i) + 3} fill="#64748b" fontSize="9" textAnchor="end">
          {d}
        </text>
      ))}
      <path d={path(truth)} fill="none" stroke="#38bdf8" strokeWidth="2" />
      <path d={path(predicted)} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4 3" />
      {predicted.map((v, i) =>
        v === null ? null : <circle key={i} cx={x(v)} cy={y(i)} r="2.5" fill="#f59e0b" />
      )}
      <text x={PAD.left} y={Hgt - 4} fill="#94a3b8" fontSize="10">
        Temperature (degC)
      </text>
    </svg>
  );
}
