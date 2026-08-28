"use client";

import { useId } from "react";
import { thermalColor, rgbCss } from "../lib/color";
import { DEPTH_MAP_MAX, LEVEL_SPAN, depthT } from "../lib/depth";

interface Props {
  depths: number[];
  predicted: (number | null)[];
  truth: (number | null)[];
  /** Faint reference curve from the first frame, so drift over time is visible. */
  reference?: (number | null)[] | null;
  /** Fixed axis across the whole series, so frames are comparable. */
  scale: { min: number; max: number };
  thermocline: number | null;
  /** Depth in metres currently selected on the depth map. */
  activeDepth: number;
}

const W = 340;
const H = 430;
const PAD = { top: 14, right: 14, bottom: 34, left: 40 };
const GUIDES = [0, 50, 100, 200, 500, 1000, 2000];

type Pt = { x: number; y: number };

/** Catmull-Rom through the samples, emitted as cubic beziers. */
function smoothPath(pts: Pt[]): string {
  if (pts.length === 0) return "";
  if (pts.length < 3) return pts.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
  let d = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
    const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
    d += ` C${c1.x.toFixed(1)},${c1.y.toFixed(1)} ${c2.x.toFixed(1)},${c2.y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
  }
  return d;
}

export default function TemporalProfileChart({
  depths,
  predicted,
  truth,
  reference,
  scale,
  thermocline,
  activeDepth,
}: Props) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");

  const has = predicted.some((v) => v !== null) || truth.some((v) => v !== null);
  if (!has) {
    return (
      <div className="flex h-[430px] items-center justify-center px-4 text-center text-sm text-slate-400">
        No column here — pick an ocean point on the Map tab.
      </div>
    );
  }

  const min = Math.floor(scale.min - 0.5);
  const max = Math.ceil(scale.max + 0.5);
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const x = (t: number) => PAD.left + ((t - min) / (max - min || 1)) * plotW;
  const y = (depth: number) => PAD.top + depthT(depth, depths) * plotH;

  const points = (series: (number | null)[]): Pt[] =>
    series
      .map((v, i) => (v === null || i >= depths.length ? null : { x: x(v), y: y(depths[i]) }))
      .filter((p): p is Pt => p !== null);

  const predPts = points(predicted);
  const truthPts = points(truth);
  const refPts = reference ? points(reference) : [];

  // Fill between the left axis and the predicted curve, tinted by depth.
  const area =
    predPts.length > 1
      ? `${smoothPath(predPts)} L${PAD.left},${predPts[predPts.length - 1].y.toFixed(1)} L${PAD.left},${predPts[0].y.toFixed(1)} Z`
      : "";

  const fillStops = depths.map((d, i) => {
    const v = predicted[i];
    return {
      offset: `${(depthT(d, depths) * 100).toFixed(2)}%`,
      color: v === null || v === undefined ? "#0a1330" : rgbCss(thermalColor(v, scale.min, scale.max)),
    };
  });

  const xTicks = 4;
  const floorY = PAD.top + LEVEL_SPAN * plotH;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      role="img"
      aria-label="Depth versus temperature profile"
    >
      <defs>
        <linearGradient id={`fill${uid}`} x1="0" y1="0" x2="0" y2="1">
          {fillStops.map((s, i) => (
            <stop key={i} offset={s.offset} stopColor={s.color} stopOpacity="0.55" />
          ))}
          <stop offset={`${LEVEL_SPAN * 100}%`} stopColor="#0a1330" stopOpacity="0.2" />
          <stop offset="100%" stopColor="#05070d" stopOpacity="0" />
        </linearGradient>
      </defs>

      {Array.from({ length: xTicks + 1 }).map((_, i) => {
        const t = min + ((max - min) * i) / xTicks;
        return (
          <g key={i}>
            <line x1={x(t)} y1={PAD.top} x2={x(t)} y2={H - PAD.bottom} stroke="#1e293b" />
            <text
              x={x(t)}
              y={H - PAD.bottom + 15}
              fill="#64748b"
              fontSize="9.5"
              textAnchor="middle"
            >
              {t.toFixed(0)}
            </text>
          </g>
        );
      })}

      {GUIDES.filter((d) => d <= DEPTH_MAP_MAX).map((d) => (
        <g key={d}>
          <line
            x1={PAD.left}
            y1={y(d)}
            x2={W - PAD.right}
            y2={y(d)}
            stroke="#1e293b"
            strokeDasharray="2 3"
          />
          <text x={PAD.left - 5} y={y(d) + 3} fill="#64748b" fontSize="8.5" textAnchor="end">
            {d}
          </text>
        </g>
      ))}

      {/* Unresolved abyssal tail. */}
      <rect
        x={PAD.left}
        y={floorY}
        width={plotW}
        height={H - PAD.bottom - floorY}
        fill="rgba(2,6,23,0.5)"
      />
      <line
        x1={PAD.left}
        y1={floorY}
        x2={W - PAD.right}
        y2={floorY}
        stroke="rgba(148,163,184,0.45)"
        strokeDasharray="3 2"
      />
      <text x={W - PAD.right - 2} y={floorY + 11} fill="#475569" fontSize="8" textAnchor="end">
        below model floor
      </text>

      {area && <path d={area} fill={`url(#fill${uid})`} stroke="none" />}

      {refPts.length > 1 && (
        <path
          d={smoothPath(refPts)}
          fill="none"
          stroke="#475569"
          strokeWidth="1.25"
          strokeDasharray="2 3"
        />
      )}
      {truthPts.length > 1 && (
        <path d={smoothPath(truthPts)} fill="none" stroke="#38bdf8" strokeWidth="2" />
      )}
      {predPts.length > 1 && (
        <path
          d={smoothPath(predPts)}
          fill="none"
          stroke="#f59e0b"
          strokeWidth="2.2"
          strokeDasharray="5 3"
        />
      )}
      {predPts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="2.2" fill="#f59e0b" />
      ))}

      {thermocline !== null && (
        <g>
          <line
            x1={PAD.left}
            y1={y(thermocline)}
            x2={W - PAD.right}
            y2={y(thermocline)}
            stroke="#22d3ee"
            strokeWidth="1.4"
            strokeDasharray="4 3"
          />
          <text
            x={W - PAD.right - 2}
            y={y(thermocline) - 4}
            fill="#22d3ee"
            fontSize="8.5"
            textAnchor="end"
          >
            thermocline {thermocline} m
          </text>
        </g>
      )}

      {/* Depth-map selection, mirrored onto the chart. */}
      <line
        x1={PAD.left}
        y1={y(activeDepth)}
        x2={W - PAD.right}
        y2={y(activeDepth)}
        stroke="#f59e0b"
        strokeWidth="1"
        opacity="0.65"
      />
      <circle cx={PAD.left} cy={y(activeDepth)} r="3" fill="#fbbf24" />

      <text x={PAD.left} y={H - 3} fill="#94a3b8" fontSize="9.5">
        Temperature (degC)
      </text>
      <text
        x={PAD.left - 33}
        y={PAD.top + plotH / 2}
        fill="#94a3b8"
        fontSize="9.5"
        textAnchor="middle"
        transform={`rotate(-90 ${PAD.left - 33} ${PAD.top + plotH / 2})`}
      >
        Depth (m)
      </text>
    </svg>
  );
}
