"use client";

import { divergingColor, rgbCss, thermalColor } from "../lib/color";

interface Props {
  mode: "thermal" | "diverging";
  min: number;
  max: number;
  absMax: number;
}

export default function Legend({ mode, min, max, absMax }: Props) {
  const steps = 24;
  const colors = Array.from({ length: steps }, (_, i) => {
    const t = i / (steps - 1);
    if (mode === "thermal") return rgbCss(thermalColor(min + t * (max - min), min, max));
    return rgbCss(divergingColor(-absMax + t * 2 * absMax, absMax));
  });
  const lo = mode === "thermal" ? min.toFixed(1) : (-absMax).toFixed(1);
  const mid = mode === "thermal" ? ((min + max) / 2).toFixed(1) : "0";
  const hi = mode === "thermal" ? max.toFixed(1) : absMax.toFixed(1);

  return (
    <div>
      <div
        className="h-3 w-full rounded"
        style={{ background: `linear-gradient(to right, ${colors.join(",")})` }}
      />
      <div className="mt-1 flex justify-between text-[10px] text-slate-400">
        <span>{lo}</span>
        <span>{mid}</span>
        <span>{hi}</span>
      </div>
    </div>
  );
}
