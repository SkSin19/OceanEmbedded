"use client";

import { useCallback, useRef } from "react";
import { thermalColor, rgbCss } from "../lib/color";
import { DEPTH_MAP_MAX, LEVEL_SPAN, depthT, levelT, nearestLevel, tToDepth } from "../lib/depth";

interface Props {
  depths: number[];
  /** Selected depth in metres. */
  value: number;
  onChange: (depth: number) => void;
  /** Column profile of the live frame, drawn as a thermal strip beside the rail. */
  profile: (number | null)[];
  scale: { min: number; max: number };
  thermocline: number | null;
}

const W = 92;
const H = 420;
const TOP = 16;
const BOT = 16;
const TRACK_X = 36;
const TRACK_W = 16;
const STRIP_X = 56;
const STRIP_W = 9;

// 0 m and 2000 m get their own end labels below.
const GUIDES = [100, 300, 500, 1000, 1500];

export default function DepthMap({
  depths,
  value,
  onChange,
  profile,
  scale,
  thermocline,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);

  const plotH = H - TOP - BOT;
  const yOf = (t: number) => TOP + t * plotH;
  const floor = depths[depths.length - 1] ?? DEPTH_MAP_MAX;

  const emitFromClientY = useCallback(
    (clientY: number) => {
      const svg = svgRef.current;
      if (!svg) return;
      const r = svg.getBoundingClientRect();
      const t = ((clientY - r.top) / r.height) * H;
      onChange(Math.round(tToDepth((t - TOP) / plotH, depths)));
    },
    [depths, onChange, plotH]
  );

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    emitFromClientY(e.clientY);
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragging.current) emitFromClientY(e.clientY);
  };
  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    dragging.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 100 : 10;
    if (e.key === "ArrowUp") {
      e.preventDefault();
      onChange(Math.max(0, value - step));
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      onChange(Math.min(DEPTH_MAP_MAX, value + step));
    } else if (e.key === "Home") {
      e.preventDefault();
      onChange(0);
    } else if (e.key === "End") {
      e.preventDefault();
      onChange(DEPTH_MAP_MAX);
    }
  };

  // The live column, as gradient stops down the rail.
  const stops = depths.map((d, i) => {
    const v = profile[i];
    return {
      offset: `${(depthT(d, depths) * 100).toFixed(2)}%`,
      color: v === null || v === undefined ? "#0a1330" : rgbCss(thermalColor(v, scale.min, scale.max)),
    };
  });

  const t = depthT(value, depths);
  const y = yOf(t);
  const belowFloor = value > floor;

  return (
    <div className="pointer-events-auto select-none rounded-xl border border-white/10 bg-slate-950/75 px-2 py-2 backdrop-blur-md">
      <div className="mb-1 px-0.5 text-[9px] font-semibold uppercase tracking-wider text-slate-500">
        Depth map
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        className="cursor-ns-resize touch-none"
        role="slider"
        tabIndex={0}
        aria-label="Depth selector"
        aria-valuemin={0}
        aria-valuemax={DEPTH_MAP_MAX}
        aria-valuenow={Math.round(value)}
        aria-valuetext={`${Math.round(value)} metres`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onKeyDown={onKeyDown}
      >
        <defs>
          <linearGradient id="dm-water" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#7dd3fc" />
            <stop offset="18%" stopColor="#38bdf8" />
            <stop offset="45%" stopColor="#1d4ed8" />
            <stop offset="72%" stopColor="#131f52" />
            <stop offset="100%" stopColor="#05070d" />
          </linearGradient>
          <linearGradient id="dm-thermal" x1="0" y1="0" x2="0" y2="1">
            {stops.map((s, i) => (
              <stop key={i} offset={s.offset} stopColor={s.color} />
            ))}
            <stop offset={`${LEVEL_SPAN * 100}%`} stopColor="#0a1330" />
            <stop offset="100%" stopColor="#05070d" />
          </linearGradient>
          <clipPath id="dm-clip">
            <rect x={TRACK_X} y={TOP} width={TRACK_W} height={plotH} rx={7} />
          </clipPath>
        </defs>

        <rect
          x={TRACK_X}
          y={TOP}
          width={TRACK_W}
          height={plotH}
          rx={7}
          fill="url(#dm-water)"
          stroke="rgba(148,163,184,0.22)"
        />

        {/* Below the deepest resolved level the model has nothing to say. */}
        <g clipPath="url(#dm-clip)">
          <rect
            x={TRACK_X}
            y={yOf(LEVEL_SPAN)}
            width={TRACK_W}
            height={plotH * (1 - LEVEL_SPAN)}
            fill="rgba(2,6,23,0.72)"
          />
          {Array.from({ length: 9 }).map((_, i) => {
            const yy = yOf(LEVEL_SPAN) + i * 13 + 5;
            return (
              <line
                key={i}
                x1={TRACK_X}
                y1={yy}
                x2={TRACK_X + TRACK_W}
                y2={yy}
                stroke="rgba(148,163,184,0.14)"
                strokeWidth="1"
              />
            );
          })}
        </g>

        {/* Live temperature strip: the same colours the 3D block is painted with. */}
        <rect
          x={STRIP_X}
          y={TOP}
          width={STRIP_W}
          height={plotH}
          rx={3}
          fill="url(#dm-thermal)"
          stroke="rgba(148,163,184,0.18)"
        />

        {/* Resolved level ticks. */}
        {depths.map((d, i) => (
          <line
            key={d}
            x1={TRACK_X - 3}
            y1={yOf(levelT(i, depths.length))}
            x2={TRACK_X}
            y2={yOf(levelT(i, depths.length))}
            stroke="rgba(148,163,184,0.5)"
            strokeWidth="1"
          />
        ))}

        {/* Metric guides. */}
        {GUIDES.map((d) => (
          <text
            key={d}
            x={TRACK_X - 6}
            y={yOf(depthT(d, depths)) + 3}
            fill="#64748b"
            fontSize="8"
            textAnchor="end"
            fontFamily="ui-monospace, monospace"
          >
            {d}
          </text>
        ))}

        <text x={TRACK_X + TRACK_W / 2} y={TOP - 5} fill="#94a3b8" fontSize="8.5" textAnchor="middle">
          0 m
        </text>
        <text
          x={TRACK_X + TRACK_W / 2}
          y={H - 5}
          fill="#94a3b8"
          fontSize="8.5"
          textAnchor="middle"
        >
          {DEPTH_MAP_MAX} m
        </text>

        {/* Model floor. */}
        <line
          x1={TRACK_X - 4}
          y1={yOf(LEVEL_SPAN)}
          x2={STRIP_X + STRIP_W + 2}
          y2={yOf(LEVEL_SPAN)}
          stroke="rgba(148,163,184,0.55)"
          strokeWidth="1"
          strokeDasharray="2 2"
        />

        {thermocline !== null && (
          <line
            x1={TRACK_X - 4}
            y1={yOf(depthT(thermocline, depths))}
            x2={STRIP_X + STRIP_W + 2}
            y2={yOf(depthT(thermocline, depths))}
            stroke="#22d3ee"
            strokeWidth="1.5"
            strokeDasharray="3 2"
            opacity="0.9"
          />
        )}

        {/* Active depth indicator. */}
        <line
          x1={TRACK_X - 8}
          y1={y}
          x2={STRIP_X + STRIP_W + 5}
          y2={y}
          stroke="#f59e0b"
          strokeWidth="1.75"
        />
        <circle cx={TRACK_X + TRACK_W / 2} cy={y} r="6.5" fill="#f59e0b" opacity="0.28" />
        <circle
          cx={TRACK_X + TRACK_W / 2}
          cy={y}
          r="3.6"
          fill="#fbbf24"
          stroke="#0b1220"
          strokeWidth="1.2"
        />
      </svg>

      <div className="mt-1 text-center">
        <div
          className={`font-mono text-[13px] font-semibold leading-tight ${
            belowFloor ? "text-slate-500" : "text-amber-400"
          }`}
        >
          {Math.round(value)} m
        </div>
        <div className="text-[9px] leading-tight text-slate-500">
          {belowFloor ? "below model floor" : `level ${depths[nearestLevel(value, depths)]} m`}
        </div>
      </div>
    </div>
  );
}
