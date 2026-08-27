// Small helpers shared by the Explorer and Playground views.

import type { Grid } from "./types";

export function gridExtent(grid: Grid): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const row of grid) {
    for (const v of row) {
      if (v === null || Number.isNaN(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (min === Infinity) return { min: 0, max: 1 };
  return { min, max };
}

export function errorGrid(pred: Grid, truth: Grid): Grid {
  return pred.map((row, r) =>
    row.map((v, c) => {
      const t = truth[r][c];
      return v === null || t === null ? null : v - t;
    })
  );
}

// Nearest ocean (non-null) cell to a target lat/lon, so a default profile point
// is not on land (the box center falls on the Indian subcontinent).
export function nearestOcean(
  grid: Grid,
  lat: number[],
  lon: number[],
  targetLat: number,
  targetLon: number
): { row: number; col: number } {
  let best = { row: 0, col: 0 };
  let bestDist = Infinity;
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[r].length; c++) {
      if (grid[r][c] === null) continue;
      const d = (lat[r] - targetLat) ** 2 + (lon[c] - targetLon) ** 2;
      if (d < bestDist) {
        bestDist = d;
        best = { row: r, col: c };
      }
    }
  }
  return best;
}
