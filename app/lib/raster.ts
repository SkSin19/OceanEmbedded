// Turn a model Grid into pixels. Shared by the Leaflet overlay (which needs an
// image URL) and the 3D volume (which needs a canvas to use as a texture).
//
// Grids are [lat][lon] with lat ascending (south -> north), so row r maps to
// display row H-1-r in every consumer here.

import type { Grid } from "./types";
import { divergingColor, thermalColor } from "./color";

export type RasterMode = "thermal" | "diverging";

export interface RasterOptions {
  mode?: RasterMode;
  min?: number;
  max?: number;
  absMax?: number;
  /** Land / no-data alpha. 0 keeps the basemap visible under the overlay. */
  landAlpha?: number;
  /** Opacity applied to valid ocean pixels. */
  alpha?: number;
}

/** Render a grid into a fresh canvas, one pixel per cell, north-up. */
export function gridToCanvas(grid: Grid, opts: RasterOptions = {}): HTMLCanvasElement {
  const {
    mode = "thermal",
    min = 0,
    max = 1,
    absMax = 1,
    landAlpha = 0,
    alpha = 255,
  } = opts;

  const H = grid.length;
  const W = grid[0]?.length ?? 0;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, W);
  canvas.height = Math.max(1, H);
  const ctx = canvas.getContext("2d");
  if (!ctx || W === 0) return canvas;

  const img = ctx.createImageData(W, H);
  for (let r = 0; r < H; r++) {
    const y = H - 1 - r; // north at the top
    for (let c = 0; c < W; c++) {
      const v = grid[r][c];
      const i = (y * W + c) * 4;
      if (v === null || Number.isNaN(v)) {
        img.data[i + 3] = landAlpha;
        continue;
      }
      const [rr, gg, bb] =
        mode === "thermal" ? thermalColor(v, min, max) : divergingColor(v, absMax);
      img.data[i] = rr;
      img.data[i + 1] = gg;
      img.data[i + 2] = bb;
      img.data[i + 3] = alpha;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Same as gridToCanvas but upscaled with smoothing, so a 51x121 model grid
 * reads as a continuous field on a zoomed-in map instead of visible blocks.
 */
export function gridToSmoothCanvas(
  grid: Grid,
  opts: RasterOptions & { scale?: number } = {}
): HTMLCanvasElement {
  const { scale = 6, ...rest } = opts;
  const src = gridToCanvas(grid, rest);
  const out = document.createElement("canvas");
  out.width = src.width * scale;
  out.height = src.height * scale;
  const ctx = out.getContext("2d");
  if (!ctx) return src;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, out.width, out.height);
  return out;
}

/** Bilinear-ish lookup of a grid value at a lat/lon, returning null over land. */
export function sampleGrid(
  grid: Grid,
  lat: number[],
  lon: number[],
  atLat: number,
  atLon: number
): number | null {
  const r = nearestIndex(lat, atLat);
  const c = nearestIndex(lon, atLon);
  const v = grid[r]?.[c];
  return v === undefined || v === null || Number.isNaN(v) ? null : v;
}

export function nearestIndex(axis: number[], value: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < axis.length; i++) {
    const d = Math.abs(axis[i] - value);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}
