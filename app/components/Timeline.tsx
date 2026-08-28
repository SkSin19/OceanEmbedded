"use client";

import { useEffect, useRef } from "react";

interface Props {
  dates: string[];
  index: number;
  onIndex: (i: number) => void;
  playing: boolean;
  onPlaying: (p: boolean) => void;
  /** Frames per second of playback. */
  fps: number;
  onFps: (f: number) => void;
  /** Which frames have arrived from the backend. */
  ready: boolean[];
  loaded: number;
}

export default function Timeline({
  dates,
  index,
  onIndex,
  playing,
  onPlaying,
  fps,
  onFps,
  ready,
  loaded,
}: Props) {
  const acc = useRef(0);
  const last = useRef(0);
  // Read through refs so the rAF loop is started once per play, not per frame.
  const state = useRef({ index, dates, ready, fps, onIndex });
  useEffect(() => {
    state.current = { index, dates, ready, fps, onIndex };
  });

  useEffect(() => {
    if (!playing || dates.length < 2) return;
    let raf = 0;
    acc.current = 0;
    last.current = performance.now();

    const tick = () => {
      raf = requestAnimationFrame(tick);
      const now = performance.now();
      acc.current += now - last.current;
      last.current = now;
      const step = 1000 / Math.max(0.5, state.current.fps);
      if (acc.current < step) return;
      acc.current = 0;

      const { index: i, dates: d, ready: r } = state.current;
      // Skip past frames that have not arrived yet rather than stalling on them.
      for (let k = 1; k <= d.length; k++) {
        const next = (i + k) % d.length;
        if (r[next]) {
          state.current.onIndex(next);
          return;
        }
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, dates.length]);

  const single = dates.length < 2;
  const pct = single ? 100 : (index / (dates.length - 1)) * 100;

  return (
    <div className="glass-panel rounded-2xl px-4 py-3">
      <div className="flex items-center gap-3">
        <button
          onClick={() => onPlaying(!playing)}
          disabled={single}
          aria-label={playing ? "Pause" : "Play"}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-400/30 transition hover:bg-cyan-500/25 disabled:opacity-30 disabled:hover:bg-cyan-500/15"
        >
          {playing ? (
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
              <rect x="6" y="5" width="4" height="14" rx="1" />
              <rect x="14" y="5" width="4" height="14" rx="1" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
              <path d="M8 5.5v13l11-6.5z" />
            </svg>
          )}
        </button>

        <div className="min-w-0 flex-1">
          <div className="mb-1.5 flex items-baseline justify-between gap-2">
            <span className="font-mono text-sm font-medium text-slate-100">
              {dates[index] ?? "--"}
            </span>
            <span className="shrink-0 font-mono text-[11px] text-slate-500">
              frame {dates.length ? index + 1 : 0} / {dates.length}
              {loaded < dates.length && (
                <span className="ml-2 text-amber-400">{loaded} loaded</span>
              )}
            </span>
          </div>

          <div className="relative">
            {/* Loaded-frame ticks sit behind the scrubber. */}
            <div className="pointer-events-none absolute inset-x-0 top-1/2 flex h-1.5 -translate-y-1/2 gap-px overflow-hidden rounded-full bg-slate-800">
              {ready.map((r, i) => (
                <span
                  key={i}
                  className={`h-full flex-1 ${r ? "bg-cyan-500/45" : "bg-transparent"}`}
                />
              ))}
            </div>
            <div
              className="pointer-events-none absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-gradient-to-r from-cyan-400/70 to-amber-400/70"
              style={{ left: 0, width: `${pct}%` }}
            />
            <input
              type="range"
              min={0}
              max={Math.max(0, dates.length - 1)}
              value={index}
              disabled={single}
              onChange={(e) => onIndex(Number(e.target.value))}
              aria-label="Timeline"
              className="relative w-full appearance-none bg-transparent accent-amber-500 [&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:mt-[-5px] [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-slate-950 [&::-webkit-slider-thumb]:bg-amber-400"
            />
          </div>
        </div>

        <label className="flex shrink-0 items-center gap-1.5 text-[11px] text-slate-400">
          <span className="hidden sm:inline">Speed</span>
          <select
            value={fps}
            onChange={(e) => onFps(Number(e.target.value))}
            className="rounded-md border border-white/10 bg-slate-900/80 px-1.5 py-1 font-mono text-[11px] text-slate-300 outline-none focus:border-cyan-500/50"
          >
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
            <option value={8}>8x</option>
          </select>
        </label>
      </div>
    </div>
  );
}
