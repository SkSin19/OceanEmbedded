"use client";

import { useEffect, useRef } from "react";
import type { Grid } from "../lib/types";
import { divergingColor, thermalColor } from "../lib/color";

interface Props {
  grid: Grid; // [lat][lon], lat ascending (south -> north)
  mode: "thermal" | "diverging";
  min: number;
  max: number;
  absMax: number;
  picked: { row: number; col: number } | null;
  onPick: (row: number, col: number) => void;
}

export default function HeatmapCanvas({
  grid,
  mode,
  min,
  max,
  absMax,
  picked,
  onPick,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const H = grid.length;
  const W = grid[0]?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || W === 0) return;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(W, H);
    for (let r = 0; r < H; r++) {
      // North (max lat) at top: display row = H - 1 - r.
      const y = H - 1 - r;
      for (let c = 0; c < W; c++) {
        const v = grid[r][c];
        const idx = (y * W + c) * 4;
        if (v === null || Number.isNaN(v)) {
          img.data[idx] = 15;
          img.data[idx + 1] = 23;
          img.data[idx + 2] = 42;
          img.data[idx + 3] = 255;
          continue;
        }
        const [rr, gg, bb] =
          mode === "thermal" ? thermalColor(v, min, max) : divergingColor(v, absMax);
        img.data[idx] = rr;
        img.data[idx + 1] = gg;
        img.data[idx + 2] = bb;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [grid, mode, min, max, absMax, H, W]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const col = Math.min(W - 1, Math.max(0, Math.floor(fx * W)));
    const yDisp = Math.min(H - 1, Math.max(0, Math.floor(fy * H)));
    onPick(H - 1 - yDisp, col);
  }

  const markerLeft = picked ? ((picked.col + 0.5) / W) * 100 : 0;
  const markerTop = picked ? ((H - 1 - picked.row + 0.5) / H) * 100 : 0;

  return (
    <div className="relative w-full overflow-hidden rounded-lg ring-1 ring-white/10">
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        className="block w-full cursor-crosshair"
        style={{ imageRendering: "auto", aspectRatio: `${W} / ${H}` }}
      />
      {picked && (
        <span
          className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(0,0,0,0.6)]"
          style={{ left: `${markerLeft}%`, top: `${markerTop}%` }}
        />
      )}
    </div>
  );
}
