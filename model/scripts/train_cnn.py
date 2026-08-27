"""Train OceanEmbedNet (CNN) and export metrics, a prediction field, and the
learned satellite embedding (PCA of the bottleneck) for the dashboard.

Run:
    python scripts/train_cnn.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from dataset_cnn import IN_CHANNELS, load_cnn_data, sample_patches
from model_cnn import OceanEmbedNet

EPOCHS = 60
STEPS = 40
BATCH = 16
PATCH = 64
LR = 1e-3


def masked_mse(pred, target, mask):
    m = mask.unsqueeze(1)
    se = ((pred - target) ** 2) * m
    return se.sum() / (m.sum() * pred.shape[1] + 1e-6)


def main() -> int:
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    data = load_cnn_data(val_days=7)
    rng = np.random.default_rng(0)
    tmean = np.array(data.stats["temp"]["per_depth_mean"], "float32")
    tstd = np.array(data.stats["temp"]["per_depth_std"], "float32")
    tstd[tstd == 0] = 1.0

    model = OceanEmbedNet(in_ch=IN_CHANNELS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for _ in range(STEPS):
            xb, yb, mb = sample_patches(data, BATCH, PATCH, rng)
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(yb).to(device)
            mb = torch.from_numpy(mb).to(device)
            opt.zero_grad()
            loss = masked_mse(model(xb), yb, mb)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:>3}  train {total/STEPS:.4f}")

    # Full-grid validation metrics in physical units.
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for t in data.val_idx:
            x = torch.from_numpy(data.x[t:t + 1]).to(device)
            p = model(x).cpu().numpy()[0]  # (15,H,W) normalized
            m = data.m[t] > 0.5
            p_phys = p * tstd[:, None, None] + tmean[:, None, None]
            y_phys = data.y[t] * tstd[:, None, None] + tmean[:, None, None]
            preds.append(p_phys[:, m].T)
            truths.append(y_phys[:, m].T)
    pred = np.concatenate(preds)
    true = np.concatenate(truths)

    per = []
    for d in range(pred.shape[1]):
        p, t = pred[:, d], true[:, d]
        per.append({
            "depth": float(data.depths[d]),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "corr": float(np.corrcoef(p, t)[0, 1]),
            "bias": float(np.mean(p - t)),
        })
    metrics = {
        "overall": {
            "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
            "corr": float(np.corrcoef(pred.ravel(), true.ravel())[0, 1]),
            "bias": float(np.mean(pred - true)),
        },
        "per_depth": per,
    }
    print(f"CNN val  RMSE {metrics['overall']['rmse']:.3f} degC  "
          f"corr {metrics['overall']['corr']:.3f}  bias {metrics['overall']['bias']:.3f}")

    ck = cfg.path("processed").parent / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "OceanEmbedNet",
                "in_ch": IN_CHANNELS, "depths": data.depths.tolist()}, ck / "cnn.pt")
    with open(cfg.path("processed") / "metrics_cnn.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(cfg.path("web_export") / "metrics_cnn.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    _export_embedding(cfg, data, model, device)
    print("Done.")
    return 0


def _export_embedding(cfg, data, model, device) -> None:
    """PCA the bottleneck of the last day to an RGB image = the learned embedding."""
    t = int(data.val_idx[-1])
    H, W = data.lat.size, data.lon.size
    with torch.no_grad():
        x = torch.from_numpy(data.x[t:t + 1]).to(device)
        emb = model.encode(x)  # (1, C, h, w)
        emb_full = F.interpolate(emb, size=(H, W), mode="bilinear", align_corners=False)
    e = emb_full.cpu().numpy()[0]  # (C,H,W)
    mask = data.m[t] > 0.5
    flat = e[:, mask].T  # (Npix, C)
    flat = flat - flat.mean(0, keepdims=True)
    # Top 3 principal components via SVD.
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    comps = flat @ vt[:3].T  # (Npix, 3)
    lo, hi = comps.min(0), comps.max(0)
    rgb = np.clip((comps - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(int)

    grid = [[None for _ in range(W)] for _ in range(H)]
    idx = np.argwhere(mask)
    for k, (r, c) in enumerate(idx):
        grid[int(r)][int(c)] = [int(rgb[k, 0]), int(rgb[k, 1]), int(rgb[k, 2])]

    out = {
        "date": data.times[t],
        "lat": [round(float(x), 3) for x in data.lat[::2]],
        "lon": [round(float(x), 3) for x in data.lon[::2]],
        "rgb": [row[::2] for row in grid[::2]],
    }
    with open(cfg.path("web_export") / "embedding_sample.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("Wrote embedding_sample.json")


if __name__ == "__main__":
    import sys
    sys.exit(main())
