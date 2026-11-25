"""Project paths and config loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


PATHS = Paths()
