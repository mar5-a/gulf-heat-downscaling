"""Join station readings, ERA5 and static features into one modelling table.

Every row is one station-hour. The target is the ERA5 error:

    residual = observed air temperature - ERA5 temperature at that spot

Models predict the residual, and a corrected temperature is ERA5 plus the
predicted residual. Predicting a correction instead of the temperature itself
keeps ERA5's weather (fronts, dust, heat waves) and lets the model focus on
what ERA5 gets wrong locally.

`add_features` is used both here and when making the map, so training and
prediction build features with exactly the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import era5, geogrid, static_features, stations
from .config import PATHS
from .static_features import LAND_COVER, N_SECTORS

FEATURE_GROUPS = {
    "weather": ["era5_t2m", "era5_dpd", "era5_wind", "era5_ssrd", "era5_tcc", "era5_dt1", "era5_dt3"],
    "time": ["hour_sin", "hour_cos", "doy_sin", "doy_cos"],
    # How the site differs from the ERA5 grid points its ERA5 value is blended
    # from: how much higher it is, and how much of the value comes from points
    # ERA5 treats as sea. Absolute height is left out on purpose: the first
    # design used it as a stand-in for "inland" and drew height contours on the map.
    "era5_cell": ["elev_minus_era5", "era5_sea_share"],
    "sea": ["dist_sea_km", "sea_frac_10km", "sea_frac_25km", "upwind_sea", "upwind_sea_x_wind"],
    "land_cover": [f"{c}_{r}" for c in LAND_COVER for r in ("near", "far")],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Features that change hour by hour. "upwind_sea" is a place feature picked by the
# hour's wind, so it varies within a station and cannot act as a station label.
HOURLY_FEATURES = FEATURE_GROUPS["weather"] + FEATURE_GROUPS["time"] + ["upwind_sea", "upwind_sea_x_wind"]
# Features fixed for a station. With few stations these can single out a
# station and let a model memorise its bias, so they are tested separately.
PLACE_FEATURES = [f for f in FEATURES if f not in HOURLY_FEATURES]

# A site higher than its ERA5 cell can only be made colder, never warmer
# (a LightGBM monotone constraint, -1 = decreasing). Across training stations
# ERA5's error falls about 0.4 C per 100 m by day and by night alike.
MONOTONE = {"elev_minus_era5": -1}

# Nested feature sets for LightGBM, each adding one group of place features.
# The one used for the map is chosen by station-grouped cross-validation on
# the training years.
_STATIC_SEA = [f for f in FEATURE_GROUPS["sea"] if f not in HOURLY_FEATURES]
FEATURE_SETS = {
    "lightgbm_hourly": HOURLY_FEATURES,
    "lightgbm_cell": HOURLY_FEATURES + FEATURE_GROUPS["era5_cell"],
    "lightgbm_sea": HOURLY_FEATURES + FEATURE_GROUPS["era5_cell"] + _STATIC_SEA,
    "lightgbm": FEATURES,
}
# The first design's map model (absolute height and height difference, no
# constraint), kept only as a reference so the 2025 test compares old and new
# design on the same data. Never a candidate for the map.
REFERENCE_SETS = {"lightgbm_first_design": HOURLY_FEATURES + ["elev", "elev_minus_era5"]}


def era5_series(cube, lats, lons, site_ids, times=None) -> pd.DataFrame:
    """ERA5 interpolated to each site, as a long (site, time) table with lags."""
    pts = era5.at_points(cube, lats, lons, times)
    keep = ["t2m", "d2m", "ssrd", "tcc", "wind_speed", "wind_from", "z_era5", "sea_point"]
    df = pts[keep].to_dataframe().reset_index()
    df["site"] = np.asarray(site_ids)[df["site"].to_numpy()]
    df = df.sort_values(["site", "time"])
    t = df.groupby("site")["t2m"]
    df["era5_dt1"] = df["t2m"] - t.shift(1)
    df["era5_dt3"] = df["t2m"] - t.shift(3)
    return df.rename(columns={"t2m": "era5_t2m", "d2m": "era5_d2m", "ssrd": "era5_ssrd",
                              "tcc": "era5_tcc", "wind_speed": "era5_wind", "sea_point": "era5_sea_share"})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. Needs ERA5 columns, static columns, `time` and `utc_offset`."""
    df = df.copy()
    df["era5_dpd"] = df["era5_t2m"] - df["era5_d2m"]          # dryness of the air
    df["elev_minus_era5"] = df["elev"] - df["z_era5"]
    local = df["time"] + pd.to_timedelta(df["utc_offset"], unit="h")
    df["local_hour"] = local.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["local_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["local_hour"] / 24)
    doy = local.dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # Share of sea in the direction the wind is coming from, within 10 km.
    width = 360 / N_SECTORS
    sector = np.floor(((df["wind_from"] + width / 2) % 360) / width).astype(int).clip(0, N_SECTORS - 1)
    sector_cols = df[[f"sea_sector_{k:02d}" for k in range(N_SECTORS)]].to_numpy()
    df["upwind_sea"] = sector_cols[np.arange(len(df)), sector.to_numpy()]
    df["upwind_sea_x_wind"] = df["upwind_sea"] * df["era5_wind"]
    df["season"] = local.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                       6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    return df


def drop_spikes(obs: pd.DataFrame, max_jump: float = 8.0) -> pd.DataFrame:
    """Remove readings far from their station's 5-hour running median."""
    obs = obs.sort_values(["station", "time"]).copy()
    med = obs.groupby("station")["temp"].transform(lambda s: s.rolling(5, center=True, min_periods=3).median())
    bad = (obs["temp"] - med).abs() > max_jump
    bad |= ~obs["temp"].between(-5, 56)
    return obs[~bad]


def station_static(meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Static features per station, cached by station and geohash cell.

    A station is recomputed only if it is new or its position changed.
    """
    path = PATHS.processed / "station_static.parquet"
    p = cfg["geohash_precision"]
    want = pd.DataFrame({"station": meta["station"].to_numpy(),
                         "geohash": [geogrid.encode(a, b, p) for a, b in zip(meta["lat"], meta["lon"])]})
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["station", "geohash"])
    static = old.merge(want, on=["station", "geohash"])
    todo = meta[~meta["station"].isin(static["station"])]
    if len(todo):
        static = pd.concat([static, static_features.station_features(todo, cfg)], ignore_index=True)
        static.to_parquet(path)
    return static


def drop_untrusted(meta: pd.DataFrame, static: pd.DataFrame, max_height_gap: float = 100.0) -> pd.DataFrame:
    """Drop stations whose position cannot be trusted, and record why.

    If the listed height is far from the terrain height at the listed spot,
    the coordinates are probably wrong, so every place feature would describe
    the wrong place. A cell that is still mostly water means no land was
    found within reach.
    """
    chk = meta[["station", "name", "country", "lat", "lon", "elev_station"]].merge(
        static[["station", "elev", "water_near"]], on="station")
    gap = (chk["elev_station"] - chk["elev"]).abs()
    reason = np.select([gap > max_height_gap, chk["water_near"] > 0.5],
                       [f"listed height more than {max_height_gap:.0f} m from terrain height",
                        "position still on water after moving to land"], "")
    dropped = chk.assign(height_gap_m=gap.round(0), reason=reason)[reason != ""]
    dropped.to_csv(PATHS.processed / "dropped_stations.csv", index=False)
    for r in dropped.itertuples():
        print(f"  dropped {r.name} ({r.country}): {r.reason}")
    return meta[~meta["station"].isin(dropped["station"])].reset_index(drop=True)


def build_table(cfg: dict) -> pd.DataFrame:
    PATHS.ensure()
    meta, obs = stations.load_observations(cfg)
    meta.to_csv(PATHS.processed / "stations.csv", index=False)
    meta = meta[meta["years_kept"] != ""].reset_index(drop=True)
    n_raw = len(obs)
    obs = drop_spikes(obs)
    print(f"  readings: {n_raw:,} kept {len(obs):,} after spike filter")

    static = station_static(meta, cfg)
    meta = drop_untrusted(meta, static)

    cube = era5.open_cube()
    ok = [era5.has_corners(cube, a, b, cfg["era5"]["grid_step"]) for a, b in zip(meta["lat"], meta["lon"])]
    for name in meta.loc[~np.array(ok), "name"]:
        print(f"  skipped {name}: ERA5 grid points around it not downloaded yet")
    meta = meta[ok].reset_index(drop=True)
    e = era5_series(cube, meta["lat"], meta["lon"], meta["station"]).rename(columns={"site": "station"})
    df = obs.rename(columns={"temp": "obs_t2m"})[["station", "time", "obs_t2m"]]
    df = df.merge(e, on=["station", "time"], how="inner")
    df = df.merge(static, on="station", how="left")
    df = df.merge(meta[["station", "name", "country", "lat", "lon", "elev_station"]], on="station")
    df["utc_offset"] = df["country"].map(stations.UTC_OFFSET)
    df = add_features(df)
    df["residual"] = df["obs_t2m"] - df["era5_t2m"]
    df["year"] = df["time"].dt.year
    df = df.dropna(subset=FEATURES + ["residual"]).reset_index(drop=True)
    df.to_parquet(PATHS.processed / "table.parquet")
    print(f"  table: {len(df):,} station-hours, {df['station'].nunique()} stations")
    return df


def load_table() -> pd.DataFrame:
    return pd.read_parquet(PATHS.processed / "table.parquet")
