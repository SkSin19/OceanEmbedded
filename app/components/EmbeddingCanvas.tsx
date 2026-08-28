"use client";

import { useEffect, useRef } from "react";
import type { RGB } from "../lib/types";

export default function EmbeddingCanvas({ rgb }: { rgb: RGB[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const H = rgb.length;
  const W = rgb[0]?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || W === 0) return;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(W, H);
    for (let r = 0; r < H; r++) {
      const y = H - 1 - r; // north up
      for (let c = 0; c < W; c++) {
        const px = rgb[r][c];
        const idx = (y * W + c) * 4;
        if (px === null) {
          img.data[idx] = 15;
          img.data[idx + 1] = 23;
          img.data[idx + 2] = 42;
        } else {
          img.data[idx] = px[0];
          img.data[idx + 1] = px[1];
          img.data[idx + 2] = px[2];
        }
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [rgb, H, W]);

  return (
    <div className="w-full overflow-hidden rounded-xl ring-1 ring-white/10">
      <canvas
        ref={canvasRef}
        className="block w-full"
        style={{ imageRendering: "auto", aspectRatio: `${W} / ${H}` }}
      />
    </div>
  );
}
