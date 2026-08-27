# Running OceanEmbed - detailed guide

The project has three parts that run independently:

1. **Data + model** (Python / PyTorch) - produces the Zarr, checkpoints, metrics.
2. **API** (FastAPI) - serves live inference on port 8010.
3. **Dashboard** (Next.js) - the UI, on port 3000/3001.

See `ARCHITECTURE.md` for how they fit together and what each model does.

---

## 0. Prerequisites

- **Python 3.13** (`py -3.13 --version`).
- **Node.js 18+** and npm (`node --version`).
- A free **Copernicus Marine** account: https://data.marine.copernicus.eu/register
- Optional GPU: an NVIDIA card (this repo was set up on a GTX 1650, 4 GB).
  Everything also runs on CPU.

---

## 1. One time setup

### 1a. Python environment (from `model/`)

```powershell
cd model
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1b. PyTorch

The default `pip install torch` gives a CPU build, which is fine for everything.
For GPU training install the CUDA build instead (matches the GTX 1650):

```powershell
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Note: pip treats `2.x+cpu` and `2.x+cu124` as the same version, so you must
`uninstall` first for the switch to take effect.

### 1c. Node dependencies (from repo root)

```powershell
cd ..
npm install
```

---

## 2. Build the data and models (from `model/`, venv active)

```powershell
copernicusmarine login              # once, stores a token
python scripts/download.py          # GLORYS (~1.9 GB) + satellite winds -> data/raw/
python scripts/harmonize.py         # -> data/processed/nio.zarr, stats.json, public/data/*
python scripts/train.py             # baseline MLP -> checkpoints/baseline.pt, metrics.json
python scripts/train_cnn.py         # CNN embedding -> checkpoints/cnn.pt, metrics_cnn.json
python scripts/validate_argo.py cnn # independent ARGO validation -> metrics_argo.json
```

- `download.py` pulls both the GLORYS subset and the satellite winds, skipping any
  file already present. `validate_argo.py` pulls gridded ARGO over OPeNDAP (cached).
- To change the region, dates, or depths, edit `config.yaml` and re-run from
  `download.py`.
- `download.py` and `harmonize.py` are only needed when the data window changes.
  Re-training is `train.py` / `train_cnn.py`.
- **Stop the API server before re-running `harmonize.py`**: on Windows the running
  server holds the Zarr open and the write fails otherwise.

If `download.py` reports an unknown dataset id (Copernicus occasionally renames):

```powershell
copernicusmarine describe --contains glo_phy_my
```

then update `copernicus.dataset_id` in `config.yaml`.

---

## 3. Run it (two terminals)

### Terminal 1 - backend (from `model/`, venv active)

```powershell
.\.venv\Scripts\python.exe -m uvicorn serve.app:app --port 8010
```

Check http://localhost:8010/health -> `{"status":"ok","dates":31}`.

**Port note:** port 8000 is blocked on this machine (Windows error 10013), so we
use 8010. If 8010 is also taken, pick another and set `NEXT_PUBLIC_API_BASE`
(see below) to match.

### Terminal 2 - frontend (from repo root)

```powershell
npm run dev
```

Open the URL it prints (http://localhost:3001 if 3000 is in use). The badge should
read **Live inference**; the date selector drives the backend.

If the backend is not running the dashboard still works from the static sample and
the badge reads **Static sample**.

---

## 4. Configuration

| What | Where |
|---|---|
| Region, dates, depths, variables, dataset id | `model/config.yaml` |
| API base URL used by the dashboard | env `NEXT_PUBLIC_API_BASE` (default `http://localhost:8010`) |
| Which origins the API allows (CORS) | `model/serve/app.py` |

To point the dashboard at a different API port, create `.env.local` in the repo
root:

```
NEXT_PUBLIC_API_BASE=http://localhost:8010
```

---

## 5. Quick checks and troubleshooting

- **Dashboard shows Static sample.** The API is not reachable. Start Terminal 1,
  confirm `/health`, check the port matches `NEXT_PUBLIC_API_BASE`.
- **`No module named torch`.** The venv is not active, or a torch reinstall was
  interrupted; re-run step 1b.
- **`error while attempting to bind ... 8000`.** Reserved port; use 8010.
- **Lint.** `npm run lint` should print nothing. The Python `model/` folder is
  excluded from ESLint.
- **CUDA not used.** `torch.cuda.is_available()` is False on a CPU build; reinstall
  with the cu124 index (step 1b). Training still works on CPU.

---

## 6. Command reference

```powershell
# data + model (from model/, venv active)
python scripts/download.py
python scripts/harmonize.py
python scripts/train.py
python scripts/train_cnn.py
python scripts/validate_argo.py cnn

# backend (from model/)
.\.venv\Scripts\python.exe -m uvicorn serve.app:app --port 8010

# frontend (from repo root)
npm run dev
npm run lint
npm run build      # production build
```
