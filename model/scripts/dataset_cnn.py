"""Spatial (image) dataset for the CNN embedding model.

Unlike the per-pixel baseline, this keeps the 2D structure so convolutions can
use spatial context (fronts, eddies). Each day is one (C, H, W) image:

  inputs  (10, H, W) = [sst, sss, ssh, u, v, lat, lon, sin_doy, cos_doy, mask]
  target  (15, H, W) = temperature at the 15 standard depths (per-depth normalized)
  mask    (H, W)     = 1 on ocean, 0 on land (loss is computed on ocean only)

Training samples random ocean patches (data augmentation + more diversity from a
short record); evaluation runs on the full grid since the net is fully
convolutional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from config import load_config

INPUT_VARS = ["sst", "sss", "ssh", "u", "v", "uwnd", "vwnd"]
# surface inputs + lat + lon + sin_doy + cos_doy + mask
IN_CHANNELS = len(INPUT_VARS) + 5
OUT_CHANNELS = 15


@dataclass
class CnnData:
    x: np.ndarray  # (T, 10, H, W)
    y: np.ndarray  # (T, 15, H, W)
    m: np.ndarray  # (T, H, W) float32 mask
    train_idx: np.ndarray
    val_idx: np.ndarray
    depths: np.ndarray
    stats: dict
    lat: np.ndarray
    lon: np.ndarray
    times: list[str]


def load_cnn_data(val_days: int = 7) -> CnnData:
    cfg = load_config()
    proc = cfg.path("processed")
    ds = xr.open_zarr(proc / "nio.zarr")
    import json

    with open(proc / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    lat = ds["lat"].values.astype("float32")
    lon = ds["lon"].values.astype("float32")
    depths = np.array(cfg.depths, dtype="float32")
    times = [str(t)[:10] for t in ds["time"].values]
    T, H, W = ds.sizes["time"], lat.size, lon.size

    surf = np.stack(
        [(ds[v].values - stats[v]["mean"]) / stats[v]["std"] for v in INPUT_VARS], axis=1
    ).astype("float32")  # (T,5,H,W)

    temp = ds["temp"].values.astype("float32")  # (T,15,H,W)
    tmean = np.array(stats["temp"]["per_depth_mean"], "float32")
    tstd = np.array(stats["temp"]["per_depth_std"], "float32")
    tstd[tstd == 0] = 1.0
    temp_n = (temp - tmean[None, :, None, None]) / tstd[None, :, None, None]

    # Mask from GLORYS fields only (winds have small coastal gaps, zero-filled).
    mask = (np.isfinite(surf[:, :5]).all(1) & np.isfinite(temp_n).all(1)).astype("float32")

    r = cfg.region
    latn = (2 * (lat - r["min_lat"]) / (r["max_lat"] - r["min_lat"]) - 1).astype("float32")
    lonn = (2 * (lon - r["min_lon"]) / (r["max_lon"] - r["min_lon"]) - 1).astype("float32")
    lat2d = np.broadcast_to(latn[:, None], (H, W))
    lon2d = np.broadcast_to(lonn[None, :], (H, W))

    doy = np.array(
        [(np.datetime64(t) - np.datetime64(t[:4] + "-01-01")).astype("timedelta64[D]").astype(int) + 1
         for t in times], "float32")
    sin_doy = np.sin(2 * np.pi * doy / 365.0)
    cos_doy = np.cos(2 * np.pi * doy / 365.0)

    n = len(INPUT_VARS)
    x = np.zeros((T, IN_CHANNELS, H, W), "float32")
    x[:, :n] = np.nan_to_num(surf)
    x[:, n] = lat2d[None]
    x[:, n + 1] = lon2d[None]
    x[:, n + 2] = sin_doy[:, None, None]
    x[:, n + 3] = cos_doy[:, None, None]
    x[:, n + 4] = mask
    y = np.nan_to_num(temp_n)

    split = T - val_days
    return CnnData(
        x=x, y=y, m=mask,
        train_idx=np.arange(0, split), val_idx=np.arange(split, T),
        depths=depths, stats=stats, lat=lat, lon=lon, times=times,
    )


def sample_patches(data: CnnData, batch: int, size: int, rng: np.random.Generator):
    """Random ocean patches from the training days -> (B,10,size,size) etc."""
    H, W = data.x.shape[2], data.x.shape[3]
    xs, ys, ms = [], [], []
    tries = 0
    while len(xs) < batch and tries < batch * 20:
        tries += 1
        t = int(rng.choice(data.train_idx))
        r0 = int(rng.integers(0, H - size + 1))
        c0 = int(rng.integers(0, W - size + 1))
        mp = data.m[t, r0:r0 + size, c0:c0 + size]
        if mp.mean() < 0.3:  # skip mostly-land patches
            continue
        xs.append(data.x[t, :, r0:r0 + size, c0:c0 + size])
        ys.append(data.y[t, :, r0:r0 + size, c0:c0 + size])
        ms.append(mp)
    return np.stack(xs), np.stack(ys), np.stack(ms)
