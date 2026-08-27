"use client";

import type { PerDepthMetric } from "../lib/types";

interface Props {
  perDepth: PerDepthMetric[];
  activeDepth: number;
}

const W = 520;
const Hgt = 240;
const PAD = { top: 16, right: 44, bottom: 34, left: 40 };

export default function SkillChart({ perDepth, activeDepth }: Props) {
  const plotW = W - PAD.left - PAD.right;
  const plotH = Hgt - PAD.top - PAD.bottom;
  const n = perDepth.length;

  const maxRmse = Math.max(...perDepth.map((d) => d.rmse)) * 1.15 || 1;
  const bandW = plotW / n;

  const xCorr = (i: number) => PAD.left + (i + 0.5) * bandW;
  const yCorr = (c: number) => PAD.top + (1 - c) * plotH; // corr in [0,1]

  const corrPath = perDepth
    .map((d, i) => `${i === 0 ? "M" : "L"}${xCorr(i).toFixed(1)},${yCorr(d.corr).toFixed(1)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${Hgt}`} className="w-full" role="img" aria-label="Skill by depth">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={PAD.left} y1={yCorr(g)} x2={W - PAD.right} y2={yCorr(g)} stroke="#1e293b" />
          <text x={W - PAD.right + 6} y={yCorr(g) + 3} fill="#34d399" fontSize="9">
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      {perDepth.map((d, i) => {
        const h = (d.rmse / maxRmse) * plotH;
        const active = d.depth === activeDepth;
        return (
          <g key={d.depth}>
            <rect
              x={PAD.left + i * bandW + bandW * 0.2}
              y={PAD.top + plotH - h}
              width={bandW * 0.6}
              height={h}
              fill={active ? "#f59e0b" : "#334155"}
            />
            <text
              x={PAD.left + i * bandW + bandW / 2}
              y={Hgt - PAD.bottom + 14}
              fill={active ? "#f59e0b" : "#64748b"}
              fontSize="8"
              textAnchor="middle"
            >
              {d.depth}
            </text>
          </g>
        );
      })}
      <path d={corrPath} fill="none" stroke="#34d399" strokeWidth="2" />
      {perDepth.map((d, i) => (
        <circle key={d.depth} cx={xCorr(i)} cy={yCorr(d.corr)} r="2.5" fill="#34d399" />
      ))}
      <text x={PAD.left} y={12} fill="#94a3b8" fontSize="10">
        RMSE bars (degC) . depth (m) on x
      </text>
    </svg>
  );
}
