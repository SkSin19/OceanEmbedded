"""Download a North Indian Ocean subset from Copernicus Marine (GLORYS).

Prerequisites (one time):
    pip install -r ../requirements.txt
    copernicusmarine login          # stores your credentials locally

Run:
    python scripts/download.py       # uses model/config.yaml

The subset is written to model/data/raw/glorys_nio_<start>_<end>.nc
Everything (region, dates, variables, dataset id) is read from config.yaml.
"""
from __future__ import annotations

import sys

from config import load_config


def main() -> int:
    try:
        import copernicusmarine
    except ImportError:
        print("copernicusmarine is not installed. Run: pip install -r requirements.txt")
        return 1

    cfg = load_config()
    r = cfg.region
    t = cfg.time
    raw_dir = cfg.path("raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"glorys_nio_{t['start']}_{t['end']}.nc".replace(":", "")
    print(f"Dataset : {cfg.dataset_id}")
    print(f"Vars    : {cfg.glorys_variables}")
    print(f"Region  : lat[{r['min_lat']},{r['max_lat']}] lon[{r['min_lon']},{r['max_lon']}]")
    print(f"Time    : {t['start']} .. {t['end']}")
    print(f"Output  : {raw_dir / out_name}")

    if (raw_dir / out_name).exists():
        print(f"GLORYS subset already present, skipping: {out_name}")
    else:
        copernicusmarine.subset(
            dataset_id=cfg.dataset_id,
            variables=cfg.glorys_variables,
            minimum_longitude=r["min_lon"],
            maximum_longitude=r["max_lon"],
            minimum_latitude=r["min_lat"],
            maximum_latitude=r["max_lat"],
            start_datetime=f"{t['start']}T00:00:00",
            end_datetime=f"{t['end']}T00:00:00",
            minimum_depth=0.0,
            # Go past 1000 m so the deepest target level is bracketed.
            maximum_depth=max(cfg.depths) + 150.0,
            output_directory=str(raw_dir),
            output_filename=out_name,
        )

    # Surface winds (separate satellite product, hourly -> averaged in harmonize).
    wind_name = f"wind_nio_{t['start']}_{t['end']}.nc"
    if (raw_dir / wind_name).exists():
        print(f"Wind subset already present, skipping: {wind_name}")
    else:
        print(f"Downloading winds: {cfg.wind_dataset_id}")
        copernicusmarine.subset(
            dataset_id=cfg.wind_dataset_id,
            variables=list(cfg.wind_inputs.values()),
            minimum_longitude=r["min_lon"],
            maximum_longitude=r["max_lon"],
            minimum_latitude=r["min_lat"],
            maximum_latitude=r["max_lat"],
            start_datetime=f"{t['start']}T00:00:00",
            end_datetime=f"{t['end']}T23:00:00",
            output_directory=str(raw_dir),
            output_filename=wind_name,
        )
    print("Done. Next: python scripts/harmonize.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
