"""Export a realistic SST map of the region from the real GLORYS data, for the
image-upload demo.

Two files are written:
  demo_sst_grayscale.png  - brightness encodes SST exactly the way the app decodes
                            an uploaded image, so uploading it reconstructs the
                            REAL subsurface (a true round trip).
  demo_sst_colored.png    - the same field with an oceanographic colour map, edge
                            to edge (no overlays). Upload it with the "SST colormap"
                            option on and the app decodes the colours back to SST.
  demo_sst_colorbar.png   - a labelled version with a colour bar + title, for slides.

Run:  python scripts/make_demo_sst.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import xarray as xr
from PIL import Image, ImageDraw

from config import load_config

# Same thermal ramp the dashboard uses.
THERMAL = np.array([
    [8, 20, 74], [26, 76, 160], [24, 149, 176], [64, 176, 120],
    [190, 200, 60], [240, 150, 40], [214, 47, 39],
], dtype=float)


def lut(norm: np.ndarray) -> np.ndarray:
    x = np.clip(norm, 0, 1) * (len(THERMAL) - 1)
    i = np.floor(x).astype(int)
    i = np.clip(i, 0, len(THERMAL) - 2)
    f = (x - i)[..., None]
    return (THERMAL[i] * (1 - f) + THERMAL[i + 1] * f).astype("uint8")


def main() -> int:
    date = sys.argv[1] if len(sys.argv) > 1 else "2008-08-15"
    cfg = load_config()
    proc = cfg.path("processed")
    ds = xr.open_zarr(proc / "nio.zarr")
    with open(proc / "stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    sst = ds["sst"].sel(time=np.datetime64(date), method="nearest").values.astype("float32")
    mask = np.isfinite(sst)
    lo = stats["sst"]["mean"] - 2 * stats["sst"]["std"]
    hi = stats["sst"]["mean"] + 2 * stats["sst"]["std"]
    norm = np.clip((sst - lo) / (hi - lo), 0, 1)

    # North up: our grid is south-first, image row 0 is the top.
    norm = np.flipud(norm)
    mask_f = np.flipud(mask)
    H, W = norm.shape
    up = 6  # upscale for a crisp slide image

    # Grayscale (uploadable, exact inverse of the app's decode).
    g = (norm * 255).astype("uint8")
    g[~mask_f] = 20  # land dark; the app re-masks to the day's ocean anyway
    Image.fromarray(g, "L").resize((W * up, H * up), Image.BICUBIC).save(
        _out(cfg, "demo_sst_grayscale.png")
    )

    # Colored, edge to edge (uploadable with thermal decode).
    rgb = lut(norm)
    rgb[~mask_f] = [70, 78, 92]
    img = Image.fromarray(rgb, "RGB").resize((W * up, H * up), Image.BICUBIC)
    img.save(_out(cfg, "demo_sst_colored.png"))

    # Labelled version for slides (colour bar + title); not for uploading.
    _add_colorbar(img.copy(), lo, hi, date).save(_out(cfg, "demo_sst_colorbar.png"))
    print("Wrote demo_sst_grayscale.png, demo_sst_colored.png, demo_sst_colorbar.png")
    return 0


def _out(cfg, name):
    d = cfg.path("processed").parent / "demo"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _add_colorbar(img: Image.Image, lo: float, hi: float, date: str) -> Image.Image:
    W, H = img.size
    bar_h = 46
    canvas = Image.new("RGB", (W, H + bar_h), (12, 16, 28))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    pad = 20
    bw = W - 2 * pad
    for x in range(bw):
        c = lut(np.array([x / bw]))[0]
        d.line([(pad + x, H + 10), (pad + x, H + 24)], fill=tuple(int(v) for v in c))
    for frac, lab in [(0.0, f"{lo:.1f}"), (0.5, f"{(lo+hi)/2:.1f}"), (1.0, f"{hi:.1f}")]:
        d.text((pad + int(frac * bw) - 8, H + 28), f"{lab} C", fill=(200, 210, 225))
    d.text((pad, H - 2), f"SST  {date}  North Indian Ocean", fill=(230, 235, 245))
    return canvas


if __name__ == "__main__":
    sys.exit(main())
