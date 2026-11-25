"""ERA5 hourly reanalysis at 0.25 degree grid points, stored as a Zarr cube.

ERA5 is fetched point by point from the Open-Meteo archive API, which serves
the ERA5 grid without an account. ``elevation=nan`` turns off Open-Meteo's own
elevation correction, so the values are the raw grid cell, plus the grid
cell's mean height. Only the grid points needed (the four around each station
and the ones covering the map) are fetched. They are written into one
(time, latitude, longitude) cube so that points can be read with ordinary
xarray interpolation.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .config import PATHS

API = "https://archive-api.open-meteo.com/v1/archive"

RENAME = {
    "temperature_2m": "t2m",
    "dew_point_2m": "d2m",
    "shortwave_radiation": "ssrd",
    "cloud_cover": "tcc",
}


def corner_nodes(lat: float, lon: float, step: float = 0.25) -> list[tuple[float, float]]:
    """The four grid points that surround a location."""
    lat0, lon0 = np.floor(lat / step) * step, np.floor(lon / step) * step
    return [(round(a, 4), round(b, 4)) for a, b in product((lat0, lat0 + step), (lon0, lon0 + step))]


def _span(cfg):
    years = cfg["train_years"] + cfg["test_years"]
    return f"{min(years)}-01-01", f"{max(years)}-12-31"


def _year_files(lat: float, lon: float) -> dict:
    """Years already downloaded for one grid point, as {year: cache file}."""
    out = {}
    for f in (PATHS.raw / "era5").glob(f"node_{lat:.2f}_{lon:.2f}_*.parquet"):
        first, last = (int(part[:4]) for part in f.stem.split("_")[3:5])
        out |= {y: f for y in range(first, last + 1)}
    return out


def missing_years(lat: float, lon: float, years) -> list[int]:
    return sorted(set(years) - set(_year_files(lat, lon)))


def _runs(years: list[int]) -> list[list[int]]:
    """Split sorted years into runs of consecutive years, one request each."""
    runs = []
    for y in years:
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    return runs


def node_series(lat: float, lon: float, years, cfg: dict, fetch: bool = True):
    """Hourly ERA5 at one grid point for the given years.

    Only years not already cached are downloaded, so adding a year to the
    study costs one year of quota per grid point, not the whole span again.
    Returns (None, nan) if nothing is cached and ``fetch`` is False.
    """
    if fetch:
        for run in _runs(missing_years(lat, lon, years)):
            fetch_node(lat, lon, f"{run[0]}-01-01", f"{run[-1]}-12-31", cfg["era5"]["variables"])
    have = _year_files(lat, lon)
    files = sorted({have[y] for y in years if y in have})
    if not files:
        return None, np.nan
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    df = df[df.index.year.isin(list(years)) & ~df.index.duplicated()]
    return df.drop(columns="z_era5"), float(df["z_era5"].iloc[0])


def fetch_node(lat: float, lon: float, start: str, end: str, variables: list[str]) -> tuple[pd.DataFrame, float]:
    """Hourly ERA5 values at one grid point, cached as parquet."""
    cache = PATHS.raw / "era5" / f"node_{lat:.2f}_{lon:.2f}_{start}_{end}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        df = pd.read_parquet(cache)
        return df.drop(columns="z_era5"), float(df["z_era5"].iloc[0])

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(variables),
        "models": "era5", "elevation": "nan", "timezone": "GMT",
    }
    # The free API counts a request as one call per 14 days of data (a year is
    # about 26 calls), against limits of 600 a minute, 5,000 an hour and 10,000
    # a day. When it says no (HTTP
    # 429), wait and try again rather than give up; the cache keeps progress.
    errors = 0
    while True:
        try:
            r = requests.get(API, params=params, timeout=180)
        except requests.RequestException as exc:
            errors += 1
            if errors > 10:
                raise
            print(f"  ERA5 {lat}, {lon}: {exc.__class__.__name__}, retrying", flush=True)
            time.sleep(20 * errors)
            continue
        if r.status_code == 200:
            break
        if r.status_code == 429:
            reason = r.json().get("reason", "") if r.headers.get("content-type", "").startswith("application/json") else ""
            wait = {"minutely": 65, "daily": 3600}.get(reason.lower().split(" ")[0], 600)
            print(f"  ERA5 {lat}, {lon}: HTTP 429 ({reason or 'rate limit'}), waiting {wait} s", flush=True)
            time.sleep(wait)
            continue
        errors += 1
        if errors > 10:
            r.raise_for_status()
        print(f"  ERA5 {lat}, {lon}: HTTP {r.status_code}, retrying", flush=True)
        time.sleep(20 * errors)
    js = r.json()
    if abs(js["latitude"] - lat) > 0.01 or abs(js["longitude"] - lon) > 0.01:
        raise ValueError(f"API snapped ({lat}, {lon}) to ({js['latitude']}, {js['longitude']})")

    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    # Wind as east (u) and north (v) parts, in m/s. Averaging directions such
    # as 350 and 10 degrees directly would give 180, the opposite way.
    speed = df.pop("wind_speed_10m") / 3.6
    direction = np.deg2rad(df.pop("wind_direction_10m"))  # where the wind blows from
    df["u10"] = -speed * np.sin(direction)
    df["v10"] = -speed * np.cos(direction)
    df = df.rename(columns=RENAME).astype("float32")
    df.assign(z_era5=js["elevation"]).to_parquet(cache)
    time.sleep(1.0)
    return df, float(js["elevation"])


def build_cube(needs: dict, cfg: dict, fetch: bool = True) -> xr.Dataset:
    """Fetch every grid point, in the order given, and write a (time, latitude, longitude) Zarr cube.

    ``needs`` maps each grid point (lat, lon) to the years it is needed for.
    Years a point is not needed for stay empty (NaN) in the cube.
    """
    start, end = _span(cfg)
    lats = sorted({n[0] for n in needs})
    lons = sorted({n[1] for n in needs})
    times = pd.date_range(start, f"{end} 23:00", freq="h")

    series, z = {}, np.full((len(lats), len(lons)), np.nan, dtype="float32")
    for k, ((lat, lon), years) in enumerate(needs.items()):
        df, elev = node_series(lat, lon, sorted(years), cfg, fetch)
        if df is None:
            continue
        z[lats.index(lat), lons.index(lon)] = elev
        series[(lat, lon)] = df.reindex(times)
        print(f"  ERA5 node {k + 1}/{len(needs)}: {lat:.2f}, {lon:.2f}", flush=True)

    out = PATHS.interim / "era5_cube.zarr"
    coords = {"time": times, "latitude": lats, "longitude": lons}
    xr.Dataset({"z_era5": (("latitude", "longitude"), z, {"long_name": "mean height of the ERA5 grid cell (m)"})},
               coords=coords,
               attrs={"source": "ERA5 via Open-Meteo archive API, raw grid values (elevation=nan)"}
               ).to_zarr(out, mode="w")
    units = {"t2m": "degC", "d2m": "degC", "u10": "m s-1", "v10": "m s-1", "ssrd": "W m-2", "tcc": "%"}
    # One variable at a time, so memory holds one full grid of one variable,
    # not all of them. Month-long time chunks over the whole small grid:
    # reading one site's full series touches few chunks, and the empty grid
    # points compress away.
    for var in next(iter(series.values())).columns:
        a = np.full((len(times), len(lats), len(lons)), np.nan, dtype="float32")
        for (lat, lon), df in series.items():
            a[:, lats.index(lat), lons.index(lon)] = df[var].to_numpy()
        da = xr.DataArray(a, dims=("time", "latitude", "longitude"), coords=coords,
                          attrs={"units": units.get(var, "")})
        da.to_dataset(name=var).to_zarr(out, mode="a",
                                        encoding={var: {"chunks": (24 * 31, len(lats), len(lons))}})
    return open_cube()


def open_cube() -> xr.Dataset:
    # chunks=None reads lazily without needing dask.
    return xr.open_zarr(PATHS.interim / "era5_cube.zarr", chunks=None)


def has_corners(ds: xr.Dataset, lat: float, lon: float, step: float = 0.25) -> bool:
    """True if all four grid points around a site are in the cube with data."""
    lat_ok, lon_ok = set(np.round(ds["latitude"].values, 4)), set(np.round(ds["longitude"].values, 4))
    for a, b in corner_nodes(lat, lon, step):
        if a not in lat_ok or b not in lon_ok or np.isnan(ds["z_era5"].sel(latitude=a, longitude=b).item()):
            return False
    return True


def at_points(ds: xr.Dataset, lats, lons, times=None) -> xr.Dataset:
    """Bilinear interpolation from the four surrounding grid points to each site."""
    if times is not None:
        ds = ds.sel(time=times)
    # 1 at grid points ERA5 treats as sea (cell height exactly 0 m). After the
    # bilinear blend this is the share of a site's ERA5 value that comes from
    # sea points: 1 at Bahrain airport, whose four grid points are all offshore.
    # In land cover, the 0 m points average 77% water and no point above 0 m
    # is more than 57% water.
    z = ds["z_era5"]
    ds = ds.assign(sea_point=xr.where(z.isnull(), np.nan, (z == 0).astype("float32")))
    pts = ds.interp(
        latitude=xr.DataArray(np.asarray(lats), dims="site"),
        longitude=xr.DataArray(np.asarray(lons), dims="site"),
        method="linear",
    )
    pts["wind_speed"] = np.hypot(pts["u10"], pts["v10"])
    # Direction the wind comes from, in degrees clockwise from north.
    pts["wind_from"] = (np.rad2deg(np.arctan2(-pts["u10"], -pts["v10"])) + 360) % 360
    return pts
