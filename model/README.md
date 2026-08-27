# OceanEmbed - ML side

Reconstructs subsurface ocean temperature (15 depth levels) from surface fields
over the North Indian Ocean, at 0.25 deg daily resolution.

```
model/
  config.yaml          # region, dates, depths, variables - edit this, not code
  requirements.txt
  scripts/
    config.py          # loads config.yaml
    download.py        # pulls a GLORYS subset from Copernicus Marine
    harmonize.py       # regrid -> 15 depths -> Zarr + stats + web sample
  data/
    raw/               # downloaded NetCDF (git-ignored)
    processed/         # nio.zarr + stats.json (git-ignored)
```

## 1. One-time setup

```bash
cd model
py -3.13 -m venv .venv
.venv\Scripts\activate            # PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Log in to Copernicus Marine

You need a free account: https://data.marine.copernicus.eu/register
Then, once, store your credentials locally:

```bash
copernicusmarine login
```

It prompts for your username and password and saves a token, so you never put
the password in code or config.

## 3. Download the data subset

The region, dates, and variables all come from `config.yaml`. The default is a
small window (Jan 2021, North Indian Ocean) that fits a 4 GB GPU / CPU.

```bash
python scripts/download.py
```

Writes `data/raw/glorys_nio_2021-01-01_2021-01-31.nc`.

If the dataset id is rejected, list current ids and update `config.yaml`:

```bash
copernicusmarine describe --contains glo_phy_my
```

## 4. Harmonize

```bash
python scripts/harmonize.py
```

Produces:
- `data/processed/nio.zarr`      - inputs (sst,sss,ssh,u,v) + target temp(15 depths)
- `data/processed/stats.json`    - normalization means/stds (incl. per-depth)
- `../public/data/meta.json`     - grid metadata for the dashboard
- `../public/data/sample_day.json` - one real day the dashboard renders now

## Notes

- GLORYS is 1/12 deg; we regrid to 0.25 deg. Temperature is interpolated onto the
  15 standard depths: 0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000 m.
- For the PoC, surface inputs are taken from GLORYS so every field is co-located.
  **Upgrading to real satellite inputs:** swap SST for a satellite L4 SST product,
  SSS for SMAP/SMOS, SSH/SLA for the CMEMS altimetry L4, and add ERA5 10 m winds,
  then regrid each to the same 0.25 deg grid in `harmonize.py`.
- To grow the training set, widen `time` in `config.yaml` and re-run steps 3-4.
```
