"""Harmonize the raw GLORYS subset into a model-ready, analysis-ready dataset.

Pipeline:
  1. open raw NetCDF
  2. regrid lat/lon to the 0.25 deg target grid
  3. interpolate temperature onto the 15 standard depth levels
  4. assemble surface INPUT channels (sst, sss, ssh, u, v) + TARGET temp(depth)
  5. write processed Zarr + normalization stats.json
  6. export a small sample JSON for the Next.js dashboard (public/data/)

Run:
    python scripts/harmonize.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import xarray as xr

from config import load_config


def _target_grid(region: dict) -> tuple[np.ndarray, np.ndarray]:
    res = region["resolution"]
    lat = np.arange(region["min_lat"], region["max_lat"] + res / 2, res)
    lon = np.arange(region["min_lon"], region["max_lon"] + res / 2, res)
    return lat, lon


def _find_raw(cfg) -> str:
    raw_dir = cfg.path("raw")
    # Prefer the file matching the configured window so a stale file is not used.
    exact = raw_dir / f"glorys_nio_{cfg.time['start']}_{cfg.time['end']}.nc"
    if exact.exists():
        return str(exact)
    files = sorted(raw_dir.glob("glorys_nio_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No raw file in {raw_dir}. Run: python scripts/download.py"
        )
    return str(files[-1])


def main() -> int:
    cfg = load_config()
    raw_path = _find_raw(cfg)
    print(f"Opening {raw_path}")
    ds = xr.open_dataset(raw_path)

    # GLORYS names: latitude, longitude, depth, time; thetao/so/zos/uo/vo.
    lat, lon = _target_grid(cfg.region)
    print(f"Regridding to 0.25 deg grid: {lat.size} lat x {lon.size} lon")
    ds = ds.interp(latitude=lat, longitude=lon, method="linear")

    depths = np.array(cfg.depths, dtype="float32")
    si = cfg.surface_inputs

    # Surface inputs (time, lat, lon).
    def surf(varname: str) -> xr.DataArray:
        da = ds[varname]
        if "depth" in da.dims:
            da = da.sel(depth=0, method="nearest").drop_vars("depth", errors="ignore")
        return da

    inputs = xr.Dataset(
        {
            "sst": surf(si["sst"]),
            "sss": surf(si["sss"]),
            "ssh": surf(si["ssh"]),
            "u": surf(si["u"]),
            "v": surf(si["v"]),
        }
    )

    # Surface winds: hourly satellite L4 -> daily mean, regridded to the same grid.
    wind_file = cfg.path("raw") / f"wind_nio_{cfg.time['start']}_{cfg.time['end']}.nc"
    wi = cfg.wind_inputs
    if wind_file.exists() and wi:
        print(f"Adding winds from {wind_file.name}")
        wds = xr.open_dataset(wind_file).resample(time="1D").mean()
        wds = wds.interp(latitude=lat, longitude=lon, method="linear")
        wds = wds.reindex(time=inputs["time"], method="nearest")
        inputs["uwnd"] = wds[wi["uwnd"]]
        inputs["vwnd"] = wds[wi["vwnd"]]
    else:
        print("No wind file found; proceeding without winds.")

    # Target: temperature interpolated to the 15 standard depths (time, depth, lat, lon).
    # GLORYS spans ~0.49 m to ~902 m here, so 0 m and 1000 m need edge extrapolation
    # (0 m from the near-surface levels ~= SST; 1000 m from the two deepest levels).
    temp = ds[cfg.target].interp(
        depth=depths, method="linear", kwargs={"fill_value": "extrapolate"}
    )
    temp.name = "temp"

    out = xr.merge([inputs, temp.to_dataset()])
    out = out.rename({"latitude": "lat", "longitude": "lon"})

    proc_dir = cfg.path("processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = proc_dir / "nio.zarr"
    print(f"Writing {zarr_path}")
    out.to_zarr(zarr_path, mode="w")

    # Surface input variables present (winds included only if downloaded).
    surface_vars = [v for v in ["sst", "sss", "ssh", "u", "v", "uwnd", "vwnd"] if v in out]

    # Normalization stats over ocean points (ignoring NaN land).
    stats = {}
    for name in surface_vars:
        arr = out[name].values
        stats[name] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr) or 1.0),
        }
    tarr = out["temp"].values
    stats["temp"] = {
        "mean": float(np.nanmean(tarr)),
        "std": float(np.nanstd(tarr) or 1.0),
        "per_depth_mean": [float(x) for x in np.nanmean(tarr, axis=(0, 2, 3))],
        "per_depth_std": [float(x) for x in np.nanstd(tarr, axis=(0, 2, 3))],
    }
    with open(proc_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {proc_dir/'stats.json'}")

    _export_web(cfg, out, depths)
    print("Done.")
    return 0


def _round(a: np.ndarray, nd: int = 2):
    """NaN-safe rounding to a nested list with null for land."""
    a = np.where(np.isnan(a), None, np.round(a, nd))
    return a.tolist()


def _export_web(cfg, out: xr.Dataset, depths: np.ndarray) -> None:
    """Write a small, real sample the dashboard can render immediately."""
    web_dir = cfg.path("web_export")
    web_dir.mkdir(parents=True, exist_ok=True)

    lat = out["lat"].values
    lon = out["lon"].values
    times = [str(t)[:10] for t in out["time"].values]

    meta = {
        "region": cfg.region,
        "depths": depths.tolist(),
        "lat": [float(lat.min()), float(lat.max())],
        "lon": [float(lon.min()), float(lon.max())],
        "shape": {"lat": int(lat.size), "lon": int(lon.size)},
        "times": times,
    }
    with open(web_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Day 0 sample: surface fields (full res) + temperature (every 2nd point) as JSON.
    day = out.isel(time=0)
    sample = {
        "date": times[0],
        "lat": [round(float(x), 3) for x in lat],
        "lon": [round(float(x), 3) for x in lon],
        "surface": {
            v: _round(day[v].values)
            for v in ["sst", "sss", "ssh", "u", "v", "uwnd", "vwnd"]
            if v in day
        },
        "depths": depths.tolist(),
        # Coarsen temperature by 2 to keep the static file light; API serves full res.
        "temp": _round(day["temp"].values[:, ::2, ::2]),
        "temp_lat": [round(float(x), 3) for x in lat[::2]],
        "temp_lon": [round(float(x), 3) for x in lon[::2]],
    }
    with open(web_dir / "sample_day.json", "w", encoding="utf-8") as f:
        json.dump(sample, f)
    print(f"Wrote web sample to {web_dir}")


if __name__ == "__main__":
    sys.exit(main())
