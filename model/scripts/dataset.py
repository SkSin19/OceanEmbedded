"""Build model-ready tensors from the harmonized Zarr.

Turns nio.zarr (physical units) into normalized per-pixel samples:
  features (9) = [sst, sss, ssh, u, v, lat_n, lon_n, sin(doy), cos(doy)]
  target  (15) = temperature at the 15 standard depths (per-depth normalized)

Land pixels (NaN in inputs or any target depth) are dropped. The train/val
split is temporal (earlier days train, last days validate) so validation is an
honest test of generalization to unseen dates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import xarray as xr

from config import load_config

INPUT_VARS = ["sst", "sss", "ssh", "u", "v", "uwnd", "vwnd"]
FEATURE_NAMES = INPUT_VARS + ["lat_n", "lon_n", "sin_doy", "cos_doy"]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class Data:
    x_train: np.ndarray  # (Ntr, 9)
    y_train: np.ndarray  # (Ntr, 15)
    x_val: np.ndarray
    y_val: np.ndarray
    depths: np.ndarray
    stats: dict
    # kept for reconstructing a full field for the dashboard
    val_time_index: int
    grid: dict


def _normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (arr - mean) / (std if std else 1.0)


def load_data(val_days: int = 7) -> Data:
    cfg = load_config()
    proc = cfg.path("processed")
    ds = xr.open_zarr(proc / "nio.zarr")
    with open(proc / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    lat = ds["lat"].values.astype("float32")
    lon = ds["lon"].values.astype("float32")
    depths = np.array(cfg.depths, dtype="float32")
    times = ds["time"].values
    T, H, W = ds.sizes["time"], lat.size, lon.size

    # Stack inputs -> (T, 5, H, W), normalized per channel.
    inp = np.stack(
        [_normalize(ds[v].values, stats[v]["mean"], stats[v]["std"]) for v in INPUT_VARS],
        axis=1,
    ).astype("float32")

    # Target temp -> (T, 15, H, W), normalized per depth.
    temp = ds["temp"].values.astype("float32")
    tmean = np.array(stats["temp"]["per_depth_mean"], dtype="float32")
    tstd = np.array(stats["temp"]["per_depth_std"], dtype="float32")
    tstd[tstd == 0] = 1.0
    temp_n = (temp - tmean[None, :, None, None]) / tstd[None, :, None, None]

    # Static per-pixel coords, normalized to [-1, 1] over the region.
    r = cfg.region
    lat_n = (2 * (lat - r["min_lat"]) / (r["max_lat"] - r["min_lat"]) - 1).astype("float32")
    lon_n = (2 * (lon - r["min_lon"]) / (r["max_lon"] - r["min_lon"]) - 1).astype("float32")
    lat2d = np.broadcast_to(lat_n[:, None], (H, W))
    lon2d = np.broadcast_to(lon_n[None, :], (H, W))

    doy = np.array(
        [(np.datetime64(t, "D") - np.datetime64(str(t)[:4] + "-01-01"))
         .astype("timedelta64[D]").astype(int) + 1 for t in times],
        dtype="float32",
    )
    sin_doy = np.sin(2 * np.pi * doy / 365.0)
    cos_doy = np.cos(2 * np.pi * doy / 365.0)

    # Valid ocean mask from the GLORYS fields + target (winds excluded: they have
    # small coastal gaps and are zero-filled below so we keep those ocean cells).
    valid = np.isfinite(inp[:, :5]).all(axis=1) & np.isfinite(temp_n).all(axis=1)  # (T,H,W)
    inp = np.nan_to_num(inp)  # fill wind gaps (and land, which the mask excludes)

    split = T - val_days

    def build(t_slice: slice):
        xs, ys = [], []
        for t in range(t_slice.start, t_slice.stop):
            m = valid[t]  # (H,W)
            feats = np.stack(
                [inp[t, c][m] for c in range(len(INPUT_VARS))]
                + [lat2d[m], lon2d[m],
                   np.full(m.sum(), sin_doy[t], "float32"),
                   np.full(m.sum(), cos_doy[t], "float32")],
                axis=1,
            )
            xs.append(feats)
            ys.append(temp_n[t][:, m].T)  # (Npix, 15)
        return np.concatenate(xs).astype("float32"), np.concatenate(ys).astype("float32")

    x_train, y_train = build(slice(0, split))
    x_val, y_val = build(slice(split, T))

    return Data(
        x_train, y_train, x_val, y_val, depths, stats,
        val_time_index=split,
        grid={"lat": lat, "lon": lon, "valid": valid, "inp": inp,
              "temp": temp, "lat2d": lat2d, "lon2d": lon2d,
              "sin_doy": sin_doy, "cos_doy": cos_doy},
    )
