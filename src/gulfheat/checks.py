"""Data checks that run before any model is trained.

Each check returns plain numbers and a list of warnings, so a person or the
report agent can read them without opening a notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PATHS
from .dataset import FEATURES


def run_checks(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(PATHS.processed / "table.parquet")
    meta = pd.read_csv(PATHS.processed / "stations.csv")
    warnings: list[str] = []

    dropped = meta[meta["years_kept"].fillna("") == ""]
    for row in dropped.itertuples():
        warnings.append(f"{row.station} {row.name}: dropped, too few hourly readings")
    untrusted = PATHS.processed / "dropped_stations.csv"
    if untrusted.exists():
        for row in pd.read_csv(untrusted).itertuples():
            warnings.append(f"{row.station} {row.name}: dropped, {row.reason} (gap {row.height_gap_m:.0f} m)")
    if "moved_km" in meta:
        for row in meta[meta["moved_km"].fillna(0) > 0].itertuples():
            warnings.append(f"{row.station} {row.name}: listed position is on water, moved "
                            f"{row.moved_km:.2f} km to the nearest land at least 200 m from water")

    per_station = df.groupby("station").agg(
        name=("name", "first"), country=("country", "first"), rows=("residual", "size"),
        mean_residual=("residual", "mean"), elev_meta=("elev_station", "first"), elev_dem=("elev", "first"),
    )
    per_station["elev_gap_m"] = (per_station["elev_meta"] - per_station["elev_dem"]).abs()
    for st, row in per_station[per_station["elev_gap_m"] > 30].iterrows():
        warnings.append(f"{st} {row['name']}: listed height {row['elev_meta']:.0f} m vs terrain "
                        f"{row['elev_dem']:.0f} m, so the listed position may be off")
    for st, row in per_station[per_station["mean_residual"].abs() > 3].iterrows():
        warnings.append(f"{st} {row['name']}: ERA5 is off by {row['mean_residual']:+.1f} C on average")

    whole = np.isclose(df["obs_t2m"] % 1, 0).mean()
    years = df.groupby("year")["station"].nunique().to_dict()
    missing = df[FEATURES].isna().sum()
    return {
        "rows": int(len(df)),
        "stations": int(df["station"].nunique()),
        "stations_per_year": {int(k): int(v) for k, v in years.items()},
        "share_whole_degree_readings": round(float(whole), 3),
        "residual_mean": round(float(df["residual"].mean()), 3),
        "residual_std": round(float(df["residual"].std()), 3),
        "residual_abs_over_8C": int((df["residual"].abs() > 8).sum()),
        "missing_feature_values": {k: int(v) for k, v in missing[missing > 0].items()},
        "per_station": per_station.round(2).reset_index().to_dict(orient="records"),
        "warnings": warnings,
    }
