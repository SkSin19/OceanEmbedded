"""FastAPI backend for OceanEmbed live inference.

Run (from model/):
    .venv\\Scripts\\activate
    uvicorn serve.app:app --reload --port 8000

Endpoints:
    GET /health
    GET /meta                          grid + depths + available dates
    GET /metrics                       validation skill metrics
    GET /prediction?date=YYYY-MM-DD    predicted vs truth field (coarsened), all depths
    GET /profile?lat=&lon=&date=       predicted vs truth profile at nearest point
"""
from __future__ import annotations

import io
import json

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from serve.inference import OceanModel

app = FastAPI(title="OceanEmbed API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MODEL = OceanModel()


def _round_grid(a: np.ndarray, step: int = 2):
    a = a[:, ::step, ::step]
    a = np.where(np.isfinite(a), np.round(a, 2), None)
    return a.tolist()


def _round2d(a: np.ndarray, step: int = 2):
    a = a[::step, ::step]
    a = np.where(np.isfinite(a), np.round(a, 2), None)
    return a.tolist()


def _coords():
    return ([round(float(x), 3) for x in MODEL.lat[::2]],
            [round(float(x), 3) for x in MODEL.lon[::2]])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "dates": len(MODEL.times)}


@app.get("/meta")
def meta() -> dict:
    return {
        "region": MODEL.cfg.region,
        "depths": MODEL.depths.tolist(),
        "lat": [float(MODEL.lat.min()), float(MODEL.lat.max())],
        "lon": [float(MODEL.lon.min()), float(MODEL.lon.max())],
        "shape": {"lat": int(MODEL.lat.size), "lon": int(MODEL.lon.size)},
        "times": MODEL.times,
    }


@app.get("/models")
def models() -> dict:
    return {"available": ["baseline"] + (["cnn"] if MODEL.cnn is not None else []),
            "default": "cnn" if MODEL.cnn is not None else "baseline"}


@app.get("/metrics")
def metrics(model: str = Query("cnn"), validation: str = Query("holdout")) -> dict:
    proc = MODEL.cfg.path("processed")
    if validation == "argo":
        path = proc / "metrics_argo.json"
    else:
        path = proc / ("metrics_cnn.json" if model == "cnn" else "metrics.json")
    if not path.exists():
        path = proc / "metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/prediction")
def prediction(date: str = Query(...), model: str = Query("cnn")) -> dict:
    try:
        r = MODEL.predict_day(date, model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "date": date,
        "model": model,
        "depths": MODEL.depths.tolist(),
        "lat": [round(float(x), 3) for x in MODEL.lat[::2]],
        "lon": [round(float(x), 3) for x in MODEL.lon[::2]],
        "predicted": _round_grid(r["predicted"]),
        "truth": _round_grid(r["truth"]),
    }


@app.get("/profile")
def profile(lat: float = Query(...), lon: float = Query(...),
            date: str = Query(...), model: str = Query("cnn")) -> dict:
    try:
        return MODEL.profile(lat, lon, date, model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/embedding")
def embedding(date: str = Query(...)) -> dict:
    try:
        return MODEL.embedding(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class ReconstructBody(BaseModel):
    date: str
    model: str = "cnn"
    sst_offset: float = 0.0
    ssh_offset: float = 0.0


def _reconstruct_response(r: dict, date: str, model: str) -> dict:
    lat, lon = _coords()
    return {
        "date": date, "model": model, "depths": MODEL.depths.tolist(),
        "lat": lat, "lon": lon,
        "predicted": _round_grid(r["predicted"]),
        "baseline": _round_grid(r["baseline"]),
        "truth": _round_grid(r["truth"]),
        "sst_input": _round2d(r["sst_input"]),
    }


@app.post("/reconstruct")
def reconstruct(body: ReconstructBody) -> dict:
    """Marine-heatwave what-if: reconstruct after shifting surface SST / SSH."""
    try:
        r = MODEL.reconstruct(body.date, body.model, body.sst_offset, body.ssh_offset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _reconstruct_response(r, body.date, body.model)


@app.post("/reconstruct/image")
async def reconstruct_image(
    file: UploadFile = File(...),
    date: str = Form(...),
    model: str = Form("cnn"),
) -> dict:
    """Feed an uploaded image as the SST field and reconstruct the subsurface.
    Grayscale luminance is mapped onto a plausible SST range; the other surface
    fields keep the chosen day's values."""
    from PIL import Image

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    H, W = MODEL.lat.size, MODEL.lon.size
    arr = np.asarray(img.resize((W, H))).astype("float32")
    arr = np.flipud(arr)  # image top -> north; grid row 0 is south
    lo = MODEL.stats["sst"]["mean"] - 2 * MODEL.stats["sst"]["std"]
    hi = MODEL.stats["sst"]["mean"] + 2 * MODEL.stats["sst"]["std"]
    sst_field = lo + (arr / 255.0) * (hi - lo)

    try:
        r = MODEL.reconstruct(date, model, sst_override=sst_field)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _reconstruct_response(r, date, model)
