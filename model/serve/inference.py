"""Load the trained models once and reconstruct fields / profiles / embeddings.

Serves both the per-pixel baseline (ProfileMLP) and the CNN embedding model
(OceanEmbedNet). Reuses config.yaml, stats.json, the harmonized Zarr, and the
checkpoints in model/checkpoints/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from config import load_config  # noqa: E402
from dataset import INPUT_VARS  # noqa: E402
from model import ProfileMLP  # noqa: E402
from model_cnn import OceanEmbedNet  # noqa: E402


class OceanModel:
    def __init__(self) -> None:
        cfg = load_config()
        self.cfg = cfg
        proc = cfg.path("processed")
        self.ds = xr.open_zarr(proc / "nio.zarr")
        with open(proc / "stats.json", "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        self.lat = self.ds["lat"].values.astype("float32")
        self.lon = self.ds["lon"].values.astype("float32")
        self.depths = np.array(cfg.depths, dtype="float32")
        self.times = [str(t)[:10] for t in self.ds["time"].values]
        H, W = self.lat.size, self.lon.size

        r = cfg.region
        latn = 2 * (self.lat - r["min_lat"]) / (r["max_lat"] - r["min_lat"]) - 1
        lonn = 2 * (self.lon - r["min_lon"]) / (r["max_lon"] - r["min_lon"]) - 1
        self.lat2d = np.broadcast_to(latn[:, None], (H, W)).astype("float32")
        self.lon2d = np.broadcast_to(lonn[None, :], (H, W)).astype("float32")

        self.tmean = np.array(self.stats["temp"]["per_depth_mean"], "float32")
        self.tstd = np.array(self.stats["temp"]["per_depth_std"], "float32")
        self.tstd[self.tstd == 0] = 1.0

        ck = proc.parent / "checkpoints"
        mlp_ck = torch.load(ck / "baseline.pt", map_location="cpu")
        self.mlp = ProfileMLP(len(mlp_ck["features"]), len(self.depths))
        self.mlp.load_state_dict(mlp_ck["state_dict"])
        self.mlp.eval()

        self.cnn = None
        cnn_path = ck / "cnn.pt"
        if cnn_path.exists():
            cnn_ck = torch.load(cnn_path, map_location="cpu")
            self.cnn = OceanEmbedNet(in_ch=cnn_ck.get("in_ch", 10))
            self.cnn.load_state_dict(cnn_ck["state_dict"])
            self.cnn.eval()

    # -- helpers -------------------------------------------------------------
    def _time_index(self, date: str) -> int:
        if date in self.times:
            return self.times.index(date)
        raise ValueError(f"date {date} not available; have {self.times[0]}..{self.times[-1]}")

    def _doy(self, date: str) -> tuple[float, float]:
        d = (np.datetime64(date, "D") - np.datetime64(date[:4] + "-01-01")).astype(int) + 1
        return float(np.sin(2 * np.pi * d / 365)), float(np.cos(2 * np.pi * d / 365))

    def _surface_norm(self, t: int):
        inp = np.stack(
            [(self.ds[v].isel(time=t).values - self.stats[v]["mean"]) / self.stats[v]["std"]
             for v in INPUT_VARS],
            axis=0,
        ).astype("float32")
        truth = self.ds["temp"].isel(time=t).values.astype("float32")
        mask = np.isfinite(inp).all(axis=0) & np.isfinite(truth).all(axis=0)
        return inp, truth, mask

    def _cnn_input(self, t: int, date: str, mask: np.ndarray, inp: np.ndarray) -> torch.Tensor:
        H, W = self.lat.size, self.lon.size
        sin_doy, cos_doy = self._doy(date)
        n = len(INPUT_VARS)
        x = np.zeros((1, n + 5, H, W), "float32")
        x[0, :n] = np.nan_to_num(inp)
        x[0, n] = self.lat2d
        x[0, n + 1] = self.lon2d
        x[0, n + 2] = sin_doy
        x[0, n + 3] = cos_doy
        x[0, n + 4] = mask.astype("float32")
        return torch.from_numpy(x)

    def _raw_physical(self, t: int):
        """Physical surface fields (NaN on land) for one day + truth + mask."""
        fields = {v: self.ds[v].isel(time=t).values.astype("float32") for v in INPUT_VARS}
        truth = self.ds["temp"].isel(time=t).values.astype("float32")
        # Mask from GLORYS fields only; winds (last two) have small coastal gaps.
        non_wind = [v for v in INPUT_VARS if v not in ("uwnd", "vwnd")]
        finite = np.stack([np.isfinite(fields[v]) for v in non_wind]).all(0)
        mask = finite & np.isfinite(truth).all(0)
        return fields, truth, mask

    def _norm_stack(self, fields: dict) -> np.ndarray:
        arr = np.stack(
            [(fields[v] - self.stats[v]["mean"]) / self.stats[v]["std"] for v in INPUT_VARS],
            axis=0,
        ).astype("float32")
        return np.nan_to_num(arr)  # fill wind gaps; land is excluded by the mask

    def _predict_from_norm(self, inp: np.ndarray, mask: np.ndarray, date: str, model: str):
        """Run the chosen model on normalized surface inputs -> (15,H,W) physical."""
        H, W = self.lat.size, self.lon.size
        if model == "cnn" and self.cnn is not None:
            with torch.no_grad():
                pn = self.cnn(self._cnn_input(0, date, mask, inp)).numpy()[0]
            field = pn * self.tstd[:, None, None] + self.tmean[:, None, None]
            field[:, ~mask] = np.nan
        else:
            sin_doy, cos_doy = self._doy(date)
            feats = np.stack(
                [inp[c][mask] for c in range(len(INPUT_VARS))]
                + [self.lat2d[mask], self.lon2d[mask],
                   np.full(mask.sum(), sin_doy, "float32"),
                   np.full(mask.sum(), cos_doy, "float32")],
                axis=1,
            ).astype("float32")
            with torch.no_grad():
                pn = self.mlp(torch.from_numpy(feats)).numpy()
            field = np.full((len(self.depths), H, W), np.nan, "float32")
            field[:, mask] = (pn * self.tstd[None] + self.tmean[None]).T
        return field

    # -- public --------------------------------------------------------------
    def predict_day(self, date: str, model: str = "cnn") -> dict:
        t = self._time_index(date)
        fields, truth, mask = self._raw_physical(t)
        field = self._predict_from_norm(self._norm_stack(fields), mask, date, model)
        return {"predicted": field, "truth": truth}

    def reconstruct(self, date: str, model: str = "cnn", sst_offset: float = 0.0,
                    ssh_offset: float = 0.0, sst_override: np.ndarray | None = None) -> dict:
        """Reconstruct after modifying the surface state: a uniform SST/SSH shift
        (a marine heatwave what-if) or a full SST field supplied as an image."""
        t = self._time_index(date)
        fields, truth, mask = self._raw_physical(t)
        base = self._predict_from_norm(self._norm_stack(fields), mask, date, model)
        if sst_override is not None:
            fields["sst"] = np.where(mask, sst_override.astype("float32"), np.nan)
        else:
            fields["sst"] = fields["sst"] + sst_offset
            fields["ssh"] = fields["ssh"] + ssh_offset
        modified = self._predict_from_norm(self._norm_stack(fields), mask, date, model)
        return {"predicted": modified, "baseline": base, "truth": truth,
                "sst_input": np.where(mask, fields["sst"], np.nan)}

    def profile(self, lat: float, lon: float, date: str, model: str = "cnn") -> dict:
        r = self.predict_day(date, model)
        ri = int(np.abs(self.lat - lat).argmin())
        ci = int(np.abs(self.lon - lon).argmin())
        clean = lambda a: [None if not np.isfinite(x) else round(float(x), 3) for x in a]
        return {
            "lat": round(float(self.lat[ri]), 3),
            "lon": round(float(self.lon[ci]), 3),
            "depths": self.depths.tolist(),
            "predicted": clean(r["predicted"][:, ri, ci]),
            "truth": clean(r["truth"][:, ri, ci]),
        }

    def surface_fields(self, date: str) -> dict:
        """The harmonized surface inputs for one day, in physical units.

        Land (and, for winds, coastal gaps) stays NaN so the client can mask it.
        """
        t = self._time_index(date)
        fields, _, mask = self._raw_physical(t)
        return {v: np.where(mask, fields[v], np.nan) for v in INPUT_VARS}

    def embedding(self, date: str) -> dict:
        """PCA of the CNN bottleneck -> RGB image of the learned latent state."""
        if self.cnn is None:
            raise ValueError("CNN model not available")
        t = self._time_index(date)
        H, W = self.lat.size, self.lon.size
        inp, _, mask = self._surface_norm(t)
        with torch.no_grad():
            emb = self.cnn.encode(self._cnn_input(t, date, mask, inp))
            emb = F.interpolate(emb, size=(H, W), mode="bilinear", align_corners=False)
        e = emb.numpy()[0]
        flat = e[:, mask].T
        flat = flat - flat.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        comps = flat @ vt[:3].T
        lo, hi = comps.min(0), comps.max(0)
        rgb = np.clip((comps - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(int)
        grid = [[None for _ in range(W)] for _ in range(H)]
        for k, (rr, cc) in enumerate(np.argwhere(mask)):
            grid[int(rr)][int(cc)] = [int(rgb[k, 0]), int(rgb[k, 1]), int(rgb[k, 2])]
        return {
            "date": date,
            "lat": [round(float(x), 3) for x in self.lat[::2]],
            "lon": [round(float(x), 3) for x in self.lon[::2]],
            "rgb": [row[::2] for row in grid[::2]],
        }
