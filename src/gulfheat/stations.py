"""Weather stations and their hourly readings, from Meteostat bulk files.

Only the observation files are used (``hourly/obs``). Meteostat's default
files fill gaps with model output, and training on model output would mean
learning to copy a model instead of the real air temperature.
"""

from __future__ import annotations

import gzip
import io
import json
import time

import pandas as pd
import requests

from .config import PATHS

STATIONS_URL = "https://bulk.meteostat.net/v2/stations/lite.json.gz"
OBS_URL = "https://bulk.meteostat.net/v2/hourly/obs/{station}.csv.gz"
OBS_COLUMNS = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow",
               "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]

# Local clock offset from UTC, used for hour-of-day breakdowns.
# Iran is 3.5 hours ahead and has had no daylight saving since 2022.
UTC_OFFSET = {"SA": 3, "KW": 3, "BH": 3, "QA": 3, "IQ": 3, "YE": 3, "AE": 4, "OM": 4, "IR": 3.5}


def _get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return r.content
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    return b""


def candidate_stations(cfg: dict) -> pd.DataFrame:
    """Stations in the study box whose hourly records span the training years."""
    cache = PATHS.raw / "meteostat_stations.json.gz"
    if not cache.exists():
        cache.write_bytes(_get(STATIONS_URL))
    stations = json.load(gzip.open(cache))

    # Records must span the training years; test-year coverage is checked per year later.
    first = f"{min(cfg['train_years'])}-01-01"
    last = f"{max(cfg['train_years'])}-12-31"
    box = cfg["region"]
    rows = []
    for s in stations:
        loc, hourly = s["location"], s["inventory"].get("hourly") or {}
        if loc["latitude"] is None or s["country"] not in cfg["countries"]:
            continue
        if not (box["lat_min"] <= loc["latitude"] <= box["lat_max"]
                and box["lon_min"] <= loc["longitude"] <= box["lon_max"]):
            continue
        if not hourly.get("start") or hourly["start"] > first or (hourly.get("end") or "") < last:
            continue
        rows.append({
            "station": s["id"],
            "name": s["name"]["en"],
            "country": s["country"],
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "elev_station": loc["elevation"],
        })
    return pd.DataFrame(rows)


def download_obs(station: str) -> pd.DataFrame:
    """All hourly observations for one station, indexed by UTC time."""
    cache = PATHS.raw / "meteostat" / f"{station}.csv.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.write_bytes(_get(OBS_URL.format(station=station)))
    df = pd.read_csv(io.BytesIO(gzip.decompress(cache.read_bytes())), names=OBS_COLUMNS)
    df["time"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    return df.set_index("time")[["temp", "dwpt", "wspd", "wdir"]]


def coverage(obs: pd.DataFrame, year: int) -> float:
    """Share of the year's hours that have a temperature reading."""
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    present = obs.loc[obs.index.year == year, "temp"].dropna()
    return len(present.index.intersection(hours)) / len(hours)


def load_observations(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download readings, keep station-years with enough coverage.

    Returns the kept stations (with a coverage column per year) and one long
    table of hourly readings.
    """
    stations = candidate_stations(cfg)
    years = cfg["train_years"] + cfg["test_years"]
    kept, frames = [], []
    for row in stations.itertuples():
        obs = download_obs(row.station)
        cov = {y: coverage(obs, y) for y in years}
        good_years = [y for y, c in cov.items() if c >= cfg["min_hourly_coverage"]]
        record = row._asdict() | {f"coverage_{y}": round(c, 3) for y, c in cov.items()}
        record.pop("Index")
        record["years_kept"] = ",".join(map(str, good_years))
        kept.append(record)
        if not good_years:
            continue
        sel = obs[obs.index.year.isin(good_years)].dropna(subset=["temp"]).copy()
        sel["station"] = row.station
        frames.append(sel.reset_index())
    meta = pd.DataFrame(kept)
    good = meta["years_kept"] != ""
    meta = pd.concat([fix_locations(meta[good]), meta[~good]], ignore_index=True)
    obs_all = pd.concat(frames, ignore_index=True)
    obs_all = obs_all.drop_duplicates(["station", "time"])
    return meta, obs_all


def fix_locations(meta: pd.DataFrame) -> pd.DataFrame:
    """Listed station positions, moved onto land where they fall on water.

    Results are cached by station and listed position, because finding the
    nearest land reads land cover around every station.
    """
    from .static_features import snap_to_land

    path = PATHS.processed / "station_locations.csv"
    cols = ["station", "lat", "lon", "listed_lat", "listed_lon", "moved_km"]
    old = pd.read_csv(path, dtype={"station": str}) if path.exists() else pd.DataFrame(columns=cols)
    key = meta[["station", "lat", "lon"]].rename(columns={"lat": "listed_lat", "lon": "listed_lon"})
    hit = key.merge(old, on=["station", "listed_lat", "listed_lon"], how="left")
    todo = meta[hit["moved_km"].isna().to_numpy()]
    parts = [hit.dropna(subset=["moved_km"])[cols]]
    if len(todo):
        parts.append(snap_to_land(todo)[cols])
    loc = pd.concat(parts, ignore_index=True).drop_duplicates("station", keep="last")
    PATHS.processed.mkdir(parents=True, exist_ok=True)
    loc.to_csv(path, index=False)
    return meta.drop(columns=["lat", "lon"]).merge(loc, on="station", how="left")
