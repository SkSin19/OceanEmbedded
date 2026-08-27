// Colormaps for the temperature heatmap. Hand-rolled so we add no dependencies.

type RGB = [number, number, number];

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function ramp(stops: RGB[], t: number): RGB {
  const x = Math.min(0.999999, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [lerp(a[0], b[0], f), lerp(a[1], b[1], f), lerp(a[2], b[2], f)];
}

// Sequential "thermal" ramp for absolute temperature (cold -> warm).
const THERMAL: RGB[] = [
  [8, 20, 74],
  [26, 76, 160],
  [24, 149, 176],
  [64, 176, 120],
  [190, 200, 60],
  [240, 150, 40],
  [214, 47, 39],
];

// Diverging ramp for error (predicted - truth), centered at 0.
const DIVERGING: RGB[] = [
  [33, 102, 172],
  [103, 169, 207],
  [230, 230, 230],
  [239, 138, 98],
  [178, 24, 43],
];

export function thermalColor(value: number, min: number, max: number): RGB {
  return ramp(THERMAL, (value - min) / (max - min || 1));
}

export function divergingColor(value: number, absMax: number): RGB {
  return ramp(DIVERGING, (value + absMax) / (2 * absMax || 1));
}

export function rgbCss([r, g, b]: RGB): string {
  return `rgb(${r}, ${g}, ${b})`;
}
