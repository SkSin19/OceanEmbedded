// Shared depth-axis maths for the temporal volume explorer.
//
// The model resolves 15 levels between 0 m and 1000 m, but the depth map is
// framed as a 0-2000 m water column. The axis therefore has two parts: the
// resolved levels occupy the top LEVEL_SPAN of the rail with even index
// spacing (the levels are already log-ish, and even spacing keeps the
// near-surface thermocline legible), and the remainder is an unresolved tail
// running down to DEPTH_MAP_MAX.

export const DEPTH_MAP_MAX = 2000;
export const LEVEL_SPAN = 0.72;

/** Axis position (0 = surface, 1 = 2000 m) of level `i`; `i` may be fractional. */
export function levelT(i: number, n: number): number {
  if (n <= 1) return 0;
  return (i / (n - 1)) * LEVEL_SPAN;
}

/** Axis position of an arbitrary depth in metres. */
export function depthT(depth: number, depths: number[]): number {
  const n = depths.length;
  if (n === 0) return 0;
  const floor = depths[n - 1];
  if (depth <= depths[0]) return 0;
  if (depth >= floor) {
    const tail = (depth - floor) / Math.max(1, DEPTH_MAP_MAX - floor);
    return LEVEL_SPAN + Math.min(1, tail) * (1 - LEVEL_SPAN);
  }
  for (let i = 1; i < n; i++) {
    if (depth <= depths[i]) {
      const f = (depth - depths[i - 1]) / (depths[i] - depths[i - 1] || 1);
      return levelT(i - 1 + f, n);
    }
  }
  return LEVEL_SPAN;
}

/** Inverse of depthT: axis position -> metres. */
export function tToDepth(t: number, depths: number[]): number {
  const n = depths.length;
  if (n === 0) return 0;
  const c = Math.min(1, Math.max(0, t));
  const floor = depths[n - 1];
  if (c >= LEVEL_SPAN) {
    const tail = (c - LEVEL_SPAN) / (1 - LEVEL_SPAN || 1);
    return floor + tail * (DEPTH_MAP_MAX - floor);
  }
  const x = (c / LEVEL_SPAN) * (n - 1);
  const i = Math.min(n - 2, Math.floor(x));
  const f = x - i;
  return depths[i] + f * (depths[i + 1] - depths[i]);
}

/** Nearest resolved level to a depth in metres. */
export function nearestLevel(depth: number, depths: number[]): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < depths.length; i++) {
    const d = Math.abs(depths[i] - depth);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** Linear interpolation of a column profile at an arbitrary depth. */
export function sampleProfile(
  profile: (number | null)[],
  depths: number[],
  depth: number
): number | null {
  const n = Math.min(profile.length, depths.length);
  if (n === 0) return null;
  if (depth <= depths[0]) return profile[0];
  if (depth >= depths[n - 1]) return profile[n - 1];
  for (let i = 1; i < n; i++) {
    if (depth <= depths[i]) {
      const a = profile[i - 1];
      const b = profile[i];
      if (a === null || b === null) return a ?? b;
      const f = (depth - depths[i - 1]) / (depths[i] - depths[i - 1] || 1);
      return a + (b - a) * f;
    }
  }
  return profile[n - 1];
}

export interface Thermocline {
  depth: number;
  gradient: number;
}

/** Steepest temperature gradient in the profile: a first-order thermocline depth. */
export function estimateThermocline(
  profile: (number | null)[],
  depths: number[]
): Thermocline | null {
  let best: Thermocline | null = null;
  for (let i = 1; i < profile.length; i++) {
    const a = profile[i - 1];
    const b = profile[i];
    if (a === null || b === null) continue;
    const dz = depths[i] - depths[i - 1];
    if (dz <= 0) continue;
    const g = (b - a) / dz;
    if (!best || g < best.gradient) {
      best = { depth: Math.round((depths[i] + depths[i - 1]) / 2), gradient: g };
    }
  }
  return best;
}

/** Extent of a set of column profiles, ignoring land/NaN gaps. */
export function profileExtent(series: (number | null)[][]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const p of series) {
    for (const v of p) {
      if (v === null || Number.isNaN(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (min === Infinity) return { min: 0, max: 1 };
  return { min, max };
}
