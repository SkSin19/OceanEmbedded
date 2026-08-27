# OceanEmbed - Architecture

Satellite embedding based deep learning framework that reconstructs the
three dimensional subsurface ocean temperature field from surface only satellite
observations, over the North Indian Ocean at 0.25 degree daily resolution.

SIH 2026, Problem 26066 (Ministry of Earth Sciences / INCOIS).

---

## 1. System overview

```
 Copernicus Marine (GLORYS)                     Python / PyTorch
 +---------------------------+   download    +-------------------------+
 |  surface + subsurface     | ------------> |  data pipeline          |
 |  reanalysis (NetCDF)      |               |  regrid, harmonize       |
 +---------------------------+               +-----------+-------------+
                                                         | nio.zarr + stats.json
                                                         v
                                             +-------------------------+
                                             |  models                 |
                                             |  - ProfileMLP (baseline)|
                                             |  - OceanEmbedNet (CNN)  |
                                             +-----------+-------------+
                                                         | checkpoints + metrics + exports
                                     +-------------------+-------------------+
                                     v                                       v
                        +------------------------+              +------------------------+
                        |  FastAPI backend       |  HTTP/JSON   |  Next.js dashboard      |
                        |  live inference        | <----------> |  map, profile, metrics  |
                        +------------------------+              +------------------------+
```

**Task.** Given the ocean surface state on a given day, estimate temperature at
15 standard depths at every 0.25 degree grid cell:

`f: (SST, SSS, SSH, U, V) -> T(z), z in {0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000} m`

The physical basis: surface fields carry indirect signatures of the subsurface
through thermocline displacement, mesoscale eddies, vertical mixing, and
ocean atmosphere coupling. A learned model recovers that mapping.

---

## 2. Data pipeline

Code: `model/scripts/` (`config.yaml`, `download.py`, `harmonize.py`).

### 2.1 Source

- **GLORYS12V1** global ocean physical reanalysis (Copernicus Marine,
  `cmems_mod_glo_phy_my_0.083deg_P1D-m`). Daily means, native 1/12 degree
  (~0.083 deg), 50 depth levels. Provides SST/SSS/SSH/currents and the
  subsurface temperature target.
- Variables pulled: `thetao` (temperature), `so` (salinity), `zos` (sea surface
  height), `uo`, `vo` (currents).
- **Surface winds** come from a separate CMEMS reprocessed L4 blended satellite
  wind product (`cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H`, hourly 0.25 deg),
  variables `eastward_wind`, `northward_wind`, averaged to daily.

This gives all six surface families the problem statement lists: SST, SSS,
SSH/SLA, surface currents (U, V), and surface winds (U, V). The GLORYS fields and
the target are co-located; the satellite winds are regridded onto the same grid
(Section 2.3). The upgrade path (Section 8) swaps the remaining GLORYS-sourced
inputs for their own satellite products.

### 2.2 Domain and grid

| Property | Value |
|---|---|
| Region | North Indian Ocean, 5N to 30N, 45E to 105E |
| Target grid | 0.25 deg x 0.25 deg -> **101 lat x 241 lon** |
| Time | daily, August 2008 (31 days), peak SW monsoon |
| Depths | 15 standard levels, 0 to 1000 m |

August 2008 is chosen because it is a strong-monsoon month that lies inside the
window where both the GLORYS reanalysis and the reprocessed L4 satellite winds
exist (the blended winds cover 1994 to 2009), and monsoon winds make the wind
input maximally informative.

Everything is driven by `config.yaml`, the single source of truth. Widening the
time window is one edit plus a re-run.

### 2.3 Harmonization steps (`harmonize.py`)

1. **Spatial regrid.** Bilinear interpolation from the 1/12 deg source onto the
   regular 0.25 deg target grid (`xarray.interp` on lat and lon). The hourly
   satellite winds are averaged to daily, then regridded onto the same grid and
   aligned in time with the GLORYS days.
2. **Depth interpolation.** Temperature is linearly interpolated from the 35
   native levels present in the subset (0.49 m to 902 m) onto the exact 15
   standard depths. The 0 m and 1000 m levels fall just outside the native range,
   so edge values use linear extrapolation (0 m from the near surface levels,
   which equals SST; 1000 m from the two deepest levels).
3. **Surface input assembly.** SST, SSS, U, V are taken at the shallowest level
   (nearest 0 m); SSH is the 2D `zos` field.
4. **Outputs.**
   - `data/processed/nio.zarr` - inputs `(sst,sss,ssh,u,v)` and target
     `temp(depth)`, dims `(time, depth, lat, lon)`.
   - `data/processed/stats.json` - per channel mean and standard deviation, and
     per depth mean and standard deviation for the target (used to normalize).
   - `public/data/meta.json`, `sample_day.json` - small real slices for the
     dashboard to render without the backend.

### 2.4 Normalization

Every input channel is standardized to zero mean and unit variance using the
statistics in `stats.json`:

`x_norm = (x - mean_c) / std_c`

The target is normalized **per depth** (each depth level has its own mean and
standard deviation) so shallow warm water and deep cold water contribute
comparably to the loss:

`T_norm(z) = (T(z) - mean_z) / std_z`

Metrics are always reported after de-normalizing back to degrees Celsius.

### 2.5 Land handling and splits

- Land cells are `NaN` in GLORYS. A per cell **ocean mask** marks cells where the
  GLORYS fields and all target depths are finite; the loss and all metrics use
  ocean cells only. The satellite winds have small coastal gaps, so they are
  excluded from the mask and zero-filled, which keeps those ocean cells.
- **Temporal hold out:** the last 7 days validate, the earlier 24 train. Because
  validation dates are never seen in training, this measures generalization to
  unseen dates rather than memorization.

---

## 3. Baseline model - ProfileMLP

Code: `model/scripts/model.py`, `dataset.py`, `train.py`.

An honest, fast baseline that the spatial models must beat. It treats every ocean
grid cell as an independent sample.

### 3.1 Features and target

Per cell per day, an 11 dimensional feature vector:

`[ sst, sss, ssh, u, v, uwnd, vwnd, lat_n, lon_n, sin(doy), cos(doy) ]`

- Surface variables (the five GLORYS fields plus the two wind components) are
  normalized (Section 2.4).
- `lat_n`, `lon_n` are the cell coordinates scaled to [-1, 1]. They let the model
  learn regional climatology.
- `sin(doy)`, `cos(doy)` encode the day of year cyclically for seasonality.

Target: the 15 normalized depth temperatures.

### 3.2 Network

```
Linear(11  -> 256) -> GELU
Linear(256 -> 256) -> GELU -> Dropout(0.1)
Linear(256 -> 128) -> GELU
Linear(128 -> 15)
```

### 3.3 Training

- Loss: mean squared error on normalized targets.
- Optimizer: Adam, learning rate 1e-3, cosine annealing.
- 80 epochs, batch 8192. About 218k training samples, 64k validation samples.
- Runs on CPU in a couple of minutes.

### 3.4 Result (temporal hold out, August 2008)

Overall validation RMSE **0.466 degC**, correlation 0.998. Error peaks in the
thermocline (75 to 150 m) where temperature varies most, which is the physically
expected behavior. The dashboard shows the full per-depth breakdown.

---

## 4. Embedding model - OceanEmbedNet (CNN)

Code: `model/scripts/model_cnn.py`, `dataset_cnn.py`, `train_cnn.py`.

The architecture the problem statement highlights: a convolutional encoder that
compresses the multi channel surface state into a compact latent
**satellite embedding**, and a decoder that expands it back into the full
subsurface temperature volume. Unlike the MLP, convolutions use spatial context
(fronts, eddies), so a cell's profile is informed by its neighbourhood.

### 4.1 Input tensor

Each day is one image of shape `(12, H, W)`:

`[ sst, sss, ssh, u, v, uwnd, vwnd, lat, lon, sin(doy), cos(doy), ocean_mask ]`

The mask channel tells the network where land is; `NaN` cells are filled with 0.
The channel count is derived from the input list, so adding an input does not
require touching the model code.

### 4.2 U-Net with a latent bottleneck

```
input (12, H, W)
  enc1  conv-block ->  32           ---------------------------+ skip
  pool /2                                                      |
  enc2  conv-block ->  64        -------------------------+ skip
  pool /2                                                 |
  enc3  conv-block -> 128     ---------------------+ skip |
  pool /2                                          |      |
  bottleneck conv-block -> 256   ==  EMBEDDING     |      |
  up /2 + concat(enc3) -> dec3 -> 128 ------------+      |
  up /2 + concat(enc2) -> dec2 ->  64 -------------------+
  up /2 + concat(enc1) -> dec1 ->  32
  1x1 conv -> 15  (temperature at 15 depths)
```

- Each `conv-block` is two `Conv2d(3x3) -> BatchNorm -> GELU` layers.
- The **bottleneck** (256 channels at 1/8 resolution) is the satellite embedding:
  a compact latent code of the surface ocean state. `model.encode(x)` returns it.
- U-Net skip connections carry high resolution detail past the bottleneck.
- Fully convolutional, so it trains on small patches and runs on the full grid.
- Inputs are reflection padded to a multiple of 8 and the output is cropped back,
  so any H, W works.

### 4.3 Training strategy

A single month has only 24 training days, which is few whole images. To get
diversity and avoid overfitting, training samples **random 64x64 ocean patches**
(patches that are at least 30 percent ocean), which turns each day into many
distinct examples. Validation runs on the full 101x241 grid.

- Loss: **masked** mean squared error, averaged over ocean cells and depths only.
- Optimizer: Adam 1e-3, cosine annealing, 60 epochs of 40 steps, batch 16.
- Trains on the GPU (GTX 1650) in a few minutes; falls back to CPU.

### 4.4 Embedding visualization

For a chosen day, the bottleneck feature map is upsampled to the full grid, the
per cell 256 dimensional vectors are reduced with PCA to 3 components, and those
are mapped to RGB. The result (`public/data/embedding_sample.json`) is a picture
of the learned latent ocean state: regions the model represents similarly share a
colour, revealing water masses and dynamic structures.

---

## 5. Backend - FastAPI

Code: `model/serve/` (`inference.py`, `app.py`).

Loads a checkpoint, the Zarr, and normalization stats once, then serves live
reconstruction. Runs on port 8010.

| Endpoint | Returns |
|---|---|
| `GET /health` | liveness and number of available dates |
| `GET /models` | available models and the default |
| `GET /meta` | region, depths, grid shape, available dates |
| `GET /metrics?model=&validation=` | skill metrics (holdout or independent ARGO) |
| `GET /prediction?date=&model=` | predicted and truth field, all depths (coarsened) |
| `GET /profile?lat=&lon=&date=&model=` | predicted and truth profile at nearest cell |
| `GET /embedding?date=` | the CNN latent embedding as an RGB image |
| `POST /reconstruct` | reconstruct after an SST / SSH shift (what-if) |
| `POST /reconstruct/image` | reconstruct using an uploaded image as the SST field |

CORS is limited to the local dashboard origins. The inference class is model
agnostic and channel-count agnostic, so a new checkpoint drops in without
changing the API.

---

## 6. Frontend - Next.js dashboard

Code: `app/` (Next.js 16 App Router, Tailwind v4, TypeScript).

Dependency free visualization (hand written canvas heatmap and SVG charts, no
chart libraries). Two tabs:

**Explorer.**
- KPI row (overall RMSE, correlation, bias, depth count).
- **Validation toggle:** GLORYS hold out vs independent ARGO (Section 7).
- Model selector (MLP baseline vs CNN embedding).
- Ocean heatmap with a depth slider and Predicted / Truth / Error / Embedding views.
- Click any ocean cell to see its vertical profile, predicted versus truth.
- Skill by depth chart (RMSE bars, correlation line).
- Date selector driving live inference, with a live / static source badge.

**Playground (judge demo).** Feed the model a modified surface state and watch the
subsurface it predicts:
- A marine-heatwave what-if (SST / SSH shift sliders), reconstructed live.
- Upload any image as the SST field and see the subsurface response.
- Shows the SST input map, the reconstruction at a chosen depth, and the
  baseline-versus-modified profile at the clicked point.

When the API is reachable the dashboard shows live per date reconstruction; when
it is not, the Explorer falls back to the static sample in `public/data/` (the
Playground needs the live backend).

---

## 7. Evaluation and honest framing

Two independent validations, both in degrees Celsius, per depth and overall:

**A. GLORYS temporal hold out** (last 7 days of the month, unseen in training):

| Model | Overall RMSE | Correlation |
|---|---|---|
| MLP baseline | 0.466 | 0.998 |
| CNN embedding | 0.378 | 0.999 |

The CNN wins by fixing the hard thermocline, where spatial context (fronts,
eddies) matters most.

**B. Independent ARGO** (IPRC/APDRC objective-analysis Argo, 1 deg monthly, derived
directly from float profiles and independent of the GLORYS training data):

| | Overall | thermocline (125 m) | deep (700 m) |
|---|---|---|---|
| CNN RMSE | 0.957 | 1.60 | 0.19 |
| Correlation | 0.991 | | |

This is the honest, harder number: ~0.96 degC against truly independent
observations, versus ~0.38 degC on the reanalysis hold out. The gap is expected,
because the model is trained on GLORYS while ARGO is a separate product with its
own error and coarser (1 deg) resolution. A correlation of 0.991 shows the
reconstructed structure tracks the independent floats closely.

- **Caveat to state plainly:** the hold-out numbers are strong partly because the
  GLORYS-sourced inputs and the target share a reanalysis and the model sees
  position. The independent ARGO comparison is the fairer measure of skill.

## 8. Upgrade path

Done in this build: all six surface input families including satellite winds, and
independent ARGO validation (Section 7). Remaining:

1. Swap the GLORYS-sourced inputs for their own satellite products: L4 SST,
   SMAP/SMOS SSS, altimetry SSH/SLA, each regridded to the grid in `harmonize.py`.
2. Wider time window (multi year, all seasons) for a fuller training set; one
   `config.yaml` edit plus a re-run.
3. Stronger architectures: Vision Transformer encoder, or attention over depth,
   reusing the same input and output contract.
```
