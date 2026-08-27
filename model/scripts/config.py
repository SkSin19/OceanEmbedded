"""Load config.yaml and resolve paths relative to the model/ directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

MODEL_DIR = Path(__file__).resolve().parents[1]  # .../model
CONFIG_PATH = MODEL_DIR / "config.yaml"


@dataclass
class Config:
    raw: dict

    @property
    def region(self) -> dict:
        return self.raw["region"]

    @property
    def time(self) -> dict:
        return self.raw["time"]

    @property
    def depths(self) -> list[float]:
        return [float(d) for d in self.raw["depths"]]

    @property
    def surface_inputs(self) -> dict:
        return self.raw["surface_inputs"]

    @property
    def target(self) -> str:
        return self.raw["target"]

    @property
    def dataset_id(self) -> str:
        return self.raw["copernicus"]["dataset_id"]

    @property
    def wind_dataset_id(self) -> str:
        return self.raw["copernicus"]["wind_dataset_id"]

    @property
    def wind_inputs(self) -> dict:
        return self.raw.get("wind_inputs", {})

    def path(self, key: str) -> Path:
        """Resolve a configured path to an absolute Path under model/."""
        return (MODEL_DIR / self.raw["paths"][key]).resolve()

    @property
    def glorys_variables(self) -> list[str]:
        """Unique GLORYS variable names needed for inputs + target."""
        names = set(self.surface_inputs.values()) | {self.target}
        return sorted(names)


def load_config() -> Config:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
