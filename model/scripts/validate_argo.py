"""Independent validation against gridded ARGO observations.

The problem statement asks for validation against independent ARGO, not just the
GLORYS reanalysis. GLORYS assimilates ARGO, so instead we compare our
reconstruction to the IPRC/APDRC objective-analysis Argo product (1 deg, monthly,
standard levels), which is derived directly from float profiles.

  1. build the monthly-mean reconstruction for the study month
  2. pull the matching month of gridded ARGO temperature (OPeNDAP), subset to the
     region, interpolate to our 15 depths, regrid to 0.25 deg
  3. report RMSE / correlation / bias per depth, on ocean cells both cover

Run (from model/, venv active, after training):
    python scripts/validate_argo.py [cnn|baseline]

The ARGO subset is cached to data/raw/argo_<month>.nc so re-runs are offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from scripts.config import load_config  # noqa: E402
from serve.inference import OceanModel  # noqa: E402

ARGO_URL = (
    "http://apdrc.soest.hawaii.edu/dods/public_data/"
    "Argo_Products/monthly_mean/Gridded_monthly_mean"
)


def _load_argo(cfg, month: str) -> xr.Dataset:
    """Return the ARGO temperature for `month`, cached locally after first pull."""
    cache = cfg.path("raw") / f"argo_{month}.nc"
    if cache.exists():
        return xr.open_dataset(cache)
    print(f"Fetching gridded ARGO for {month} from APDRC (OPeNDAP)...")
    r = cfg.region
    ds = xr.open_dataset(ARGO_URL, decode_times=True)
    sub = ds[["temp"]].sel(
        time=np.datetime64(f"{month}-15"), method="nearest"
    ).sel(
        lat=slice(r["min_lat"], r["max_lat"]),
        lon=slice(r["min_lon"], r["max_lon"]),
    )
    sub = sub.load()
    sub.to_netcdf(cache)
    print(f"Cached {cache}")
    return sub


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    cfg = load_config()
    month = cfg.time["start"][:7]  # YYYY-MM

    model = OceanModel()
    depths = model.depths

    # 1. Monthly-mean reconstruction over the study month.
    fields = [model.predict_day(d, model_name)["predicted"] for d in model.times]
    recon = np.nanmean(np.stack(fields), axis=0)  # (15, H, W)

    # 2. ARGO for the month, onto our depths and grid.
    argo = _load_argo(cfg, month)
    temp = argo["temp"]
    lev_name = "lev" if "lev" in temp.dims else "depth"
    argo_t = (
        temp.rename({lev_name: "depth"})
        .interp(depth=depths, method="linear")
        .interp(lat=model.lat, lon=model.lon, method="linear")
    )
    argo_arr = argo_t.transpose("depth", "lat", "lon").values.astype("float32")

    # 3. Compare where both are finite.
    per = []
    for d in range(len(depths)):
        p, a = recon[d], argo_arr[d]
        m = np.isfinite(p) & np.isfinite(a)
        if m.sum() < 10:
            per.append({"depth": float(depths[d]), "rmse": None, "corr": None, "bias": None})
            continue
        pv, av = p[m], a[m]
        per.append({
            "depth": float(depths[d]),
            "rmse": float(np.sqrt(np.mean((pv - av) ** 2))),
            "corr": float(np.corrcoef(pv, av)[0, 1]) if pv.std() > 1e-6 else None,
            "bias": float(np.mean(pv - av)),
        })

    allp = np.concatenate([recon[d][np.isfinite(recon[d]) & np.isfinite(argo_arr[d])]
                           for d in range(len(depths))])
    alla = np.concatenate([argo_arr[d][np.isfinite(recon[d]) & np.isfinite(argo_arr[d])]
                           for d in range(len(depths))])
    overall = {
        "rmse": float(np.sqrt(np.mean((allp - alla) ** 2))),
        "corr": float(np.corrcoef(allp, alla)[0, 1]),
        "bias": float(np.mean(allp - alla)),
    }
    metrics = {"model": model_name, "month": month, "source": "IPRC/APDRC gridded ARGO",
               "overall": overall, "per_depth": per}

    print(f"Independent ARGO validation ({model_name}, {month}):")
    print(f"  overall RMSE {overall['rmse']:.3f} degC  corr {overall['corr']:.3f}  "
          f"bias {overall['bias']:+.3f}")
    for r in per:
        if r["rmse"] is not None:
            print(f"  {r['depth']:>5.0f} m  rmse {r['rmse']:.3f}  bias {r['bias']:+.3f}")

    with open(cfg.path("processed") / "metrics_argo.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(cfg.path("web_export") / "metrics_argo.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    print("Wrote metrics_argo.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
