"""Train the baseline model and produce real skill metrics + dashboard exports.

Run:
    python scripts/train.py            # uses config.yaml + data/processed/nio.zarr

Outputs:
    checkpoints/baseline.pt            # weights + arch/feature spec
    data/processed/metrics.json        # per-depth RMSE, correlation, bias (val set)
    ../public/data/metrics.json        # same, for the dashboard
    ../public/data/prediction_sample.json  # predicted vs true field, one val day
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import load_config
from dataset import FEATURE_NAMES, INPUT_VARS, N_FEATURES, load_data
from model import ProfileMLP

EPOCHS = 80
BATCH = 8192
LR = 1e-3


def _metrics(pred: np.ndarray, true: np.ndarray, depths) -> dict:
    """Per-depth RMSE, Pearson correlation, bias in physical units (degC)."""
    per = []
    for d in range(pred.shape[1]):
        p, t = pred[:, d], true[:, d]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        bias = float(np.mean(p - t))
        if p.std() > 1e-6 and t.std() > 1e-6:
            corr = float(np.corrcoef(p, t)[0, 1])
        else:
            corr = float("nan")
        per.append({"depth": float(depths[d]), "rmse": rmse, "corr": corr, "bias": bias})
    overall = {
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "bias": float(np.mean(pred - true)),
        "corr": float(np.corrcoef(pred.ravel(), true.ravel())[0, 1]),
    }
    return {"overall": overall, "per_depth": per}


def main() -> int:
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    data = load_data(val_days=7)
    print(f"Train samples: {data.x_train.shape[0]:,}  Val samples: {data.x_val.shape[0]:,}")

    tmean = np.array(data.stats["temp"]["per_depth_mean"], dtype="float32")
    tstd = np.array(data.stats["temp"]["per_depth_std"], dtype="float32")
    tstd[tstd == 0] = 1.0

    xtr = torch.from_numpy(data.x_train)
    ytr = torch.from_numpy(data.y_train)
    xva = torch.from_numpy(data.x_val).to(device)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True)

    model = ProfileMLP(N_FEATURES, len(data.depths)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vpred = model(xva).cpu().numpy()
            vloss = float(np.mean((vpred - data.y_val) ** 2))
            print(f"epoch {epoch:>3}  train {total/len(xtr):.4f}  val {vloss:.4f}")

    # Final validation metrics in physical units.
    model.eval()
    with torch.no_grad():
        vpred_n = model(xva).cpu().numpy()
    pred = vpred_n * tstd[None] + tmean[None]
    true = data.y_val * tstd[None] + tmean[None]
    metrics = _metrics(pred, true, data.depths)
    print(f"Overall val  RMSE {metrics['overall']['rmse']:.3f} degC  "
          f"corr {metrics['overall']['corr']:.3f}  bias {metrics['overall']['bias']:.3f}")

    # Save checkpoint + metrics.
    ck = cfg.path("processed").parent / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "features": FEATURE_NAMES,
         "depths": data.depths.tolist(), "arch": "ProfileMLP"},
        ck / "baseline.pt",
    )
    proc = cfg.path("processed")
    with open(proc / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    web = cfg.path("web_export")
    web.mkdir(parents=True, exist_ok=True)
    with open(web / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    _export_prediction(cfg, data, model, device, tmean, tstd)
    print("Done.")
    return 0


def _export_prediction(cfg, data, model, device, tmean, tstd) -> None:
    """Reconstruct a full field for one validation day: predicted vs true."""
    g = data.grid
    t = data.val_time_index  # first held-out day
    H, W = g["lat"].size, g["lon"].size
    mask = g["valid"][t]

    feats = np.stack(
        [g["inp"][t, c][mask] for c in range(len(INPUT_VARS))]
        + [g["lat2d"][mask], g["lon2d"][mask],
           np.full(mask.sum(), g["sin_doy"][t], "float32"),
           np.full(mask.sum(), g["cos_doy"][t], "float32")],
        axis=1,
    ).astype("float32")

    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(feats).to(device)).cpu().numpy()
    p = p * tstd[None] + tmean[None]  # (Npix, 15) physical

    depths = data.depths
    pred_field = np.full((len(depths), H, W), np.nan, "float32")
    pred_field[:, mask] = p.T
    true_field = g["temp"][t]  # (15,H,W) physical, NaN on land

    def rnd(a, step=2):
        a = a[:, ::step, ::step]
        a = np.where(np.isnan(a), None, np.round(a, 2))
        return a.tolist()

    out = {
        "date": str(cfg.time["end"]),
        "depths": depths.tolist(),
        "lat": [round(float(x), 3) for x in g["lat"][::2]],
        "lon": [round(float(x), 3) for x in g["lon"][::2]],
        "predicted": rnd(pred_field),
        "truth": rnd(true_field),
    }
    with open(cfg.path("web_export") / "prediction_sample.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("Wrote prediction_sample.json")


if __name__ == "__main__":
    import sys
    sys.exit(main())
