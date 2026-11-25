"""Terrain, land cover and sea features on the geohash grid.

Sources, all public cloud-optimised GeoTIFFs read over HTTPS (only the
needed window is downloaded):
  * ESA WorldCover 2021, 10 m land cover, 3 x 3 degree tiles
  * Copernicus DEM GLO-30, 30 m terrain height, 1 x 1 degree tiles

Both are in EPSG:4326, the same latitude/longitude system as the geohash grid,
so "regridding" means averaging source pixels into each 150 m cell. Land cover
classes are first turned into 0/1 masks, then averaged, which gives the share
of each cell covered by that class.

Sea features use a coarser 275 m grid over a wider window, because the sea can
be tens of kilometres away and a 10 m read of that area would be about 200
million pixels.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from scipy import ndimage, signal

from . import geogrid

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

CRS = "EPSG:4326"
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif")

# WorldCover codes grouped into the classes that matter for near-surface heat.
LAND_COVER = {
    "built": [50],
    "tree": [10, 95],          # trees and mangroves
    "shrub_grass": [20, 30],
    "crop": [40],
    "bare": [60],              # sand, rock, sparse vegetation
    "water": [80, 90],         # open water and wetland
}
WATER_CODE = 80
MAX_SEA_KM = 60.0
N_SECTORS = 16
SECTOR_RADIUS_KM = 10.0


# ---------------------------------------------------------------- tile lookup

def _worldcover_tiles(bounds):
    w, s, e, n = bounds
    lats = range(int(np.floor(s / 3) * 3), int(np.floor(n / 3) * 3) + 1, 3)
    lons = range(int(np.floor(w / 3) * 3), int(np.floor(e / 3) * 3) + 1, 3)
    return [WORLDCOVER.format(tile=f"{'N' if a >= 0 else 'S'}{abs(a):02d}{'E' if b >= 0 else 'W'}{abs(b):03d}")
            for a in lats for b in lons]


def _dem_tiles(bounds):
    w, s, e, n = bounds
    out = []
    for a in range(int(np.floor(s)), int(np.floor(n)) + 1):
        for b in range(int(np.floor(w)), int(np.floor(e)) + 1):
            tile = f"{'N' if a >= 0 else 'S'}{abs(a):02d}_00_{'E' if b >= 0 else 'W'}{abs(b):03d}_00"
            out.append(DEM.format(tile=tile))
    return out


def _window(src, bounds):
    """Pixel window of a tile covering `bounds`, or None if they do not overlap."""
    w, s, e, n = bounds
    tw, ts, te, tn = src.bounds
    w, s, e, n = max(w, tw), max(s, ts), min(e, te), min(n, tn)
    if w >= e or s >= n:
        return None
    win = from_bounds(w, s, e, n, transform=src.transform)
    col0, row0 = int(np.floor(win.col_off)), int(np.floor(win.row_off))
    col1 = min(src.width, int(np.ceil(win.col_off + win.width)) + 1)
    row1 = min(src.height, int(np.ceil(win.row_off + win.height)) + 1)
    return Window(col0, row0, col1 - col0, row1 - row0)


# ---------------------------------------------------- fine grid (150 m cells)

def _average_onto(dst_shape, dst_transform, tiles, bounds, to_layers, fill_missing):
    """Average source pixels into destination cells, tile by tile.

    `to_layers` turns a raw tile array into named float layers (NaN = no
    data). Tiles that do not exist (open sea) get `fill_missing` values.
    """
    out = {}
    for url in tiles:
        try:
            src = rasterio.open(url)
        except RasterioIOError:
            layers = fill_missing
            for name, value in layers.items():
                acc = out.setdefault(name, np.full(dst_shape, np.nan, dtype="float32"))
                # A missing tile only fills cells inside that tile's square.
                acc[np.isnan(acc) & _cells_in_tile(url, dst_shape, dst_transform)] = value
            continue
        with src:
            win = _window(src, bounds)
            if win is None:
                continue
            raw = src.read(1, window=win)
            win_transform = src.window_transform(win)
            for name, layer in to_layers(raw, src.nodata).items():
                dst = np.full(dst_shape, np.nan, dtype="float32")
                reproject(layer.astype("float32"), dst,
                          src_transform=win_transform, src_crs=CRS,
                          dst_transform=dst_transform, dst_crs=CRS,
                          src_nodata=np.nan, dst_nodata=np.nan,
                          resampling=Resampling.average)
                acc = out.setdefault(name, np.full(dst_shape, np.nan, dtype="float32"))
                take = np.isnan(acc) & ~np.isnan(dst)
                acc[take] = dst[take]
    return out


def _cells_in_tile(url, shape, transform):
    """Mask of destination cells whose centres fall inside a tile's square."""
    name = url.rsplit("/", 1)[-1]
    rows, cols = np.indices(shape)
    lons, lats = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    lons, lats = np.asarray(lons).reshape(shape), np.asarray(lats).reshape(shape)
    if "WorldCover" in name:
        code = name.split("_")[5]          # e.g. N24E048
        lat0 = int(code[1:3]) * (1 if code[0] == "N" else -1)
        lon0 = int(code[4:7]) * (1 if code[3] == "E" else -1)
        size = 3
    else:
        parts = name.split("_")            # Copernicus_DSM_COG_10_N26_00_E050_00_DEM.tif
        lat0 = int(parts[4][1:]) * (1 if parts[4][0] == "N" else -1)
        lon0 = int(parts[6][1:]) * (1 if parts[6][0] == "E" else -1)
        size = 1
    return (lats >= lat0) & (lats < lat0 + size) & (lons >= lon0) & (lons < lon0 + size)


def _land_cover_layers(raw, nodata):
    valid = raw != 0
    layers = {}
    for name, codes in LAND_COVER.items():
        layer = np.isin(raw, codes).astype("float32")
        layer[~valid] = np.nan
        layers[name] = layer
    return layers


def _dem_layers(raw, nodata):
    layer = raw.astype("float32")
    if nodata is not None:
        layer[raw == nodata] = np.nan
    return {"elev": layer}


def fine_grid_features(lats, lons, bounds, near_r: int, far_r: int) -> dict[str, np.ndarray]:
    """Per-cell terrain height and land cover shares in two neighbourhoods."""
    shape = (len(lats), len(lons))
    transform = transform_from_bounds(*bounds, width=shape[1], height=shape[0])
    lc = _average_onto(shape, transform, _worldcover_tiles(bounds), bounds, _land_cover_layers,
                       fill_missing={k: (1.0 if k == "water" else 0.0) for k in LAND_COVER})
    dem = _average_onto(shape, transform, _dem_tiles(bounds), bounds, _dem_layers,
                        fill_missing={"elev": 0.0})
    feats = {"elev": dem["elev"]}
    # Neighbourhood shares are Gaussian-weighted, reaching out r cells (sigma = r / 2).
    # A plain square average turns one isolated tree cell into an 11 x 11 square of
    # equal values, and a tree model's thresholds then print those squares on the map.
    for name, frac in lc.items():
        frac = np.nan_to_num(frac, nan=0.0)
        for label, r in (("near", near_r), ("far", far_r)):
            feats[f"{name}_{label}"] = ndimage.gaussian_filter(frac, sigma=r / 2, truncate=2.0, mode="nearest")
    return feats


# -------------------------------------------------- coarse grid (sea features)

def _water_mask(bounds, pixel_deg):
    """Large water bodies on a coarse lat/lon grid, plus that grid's transform."""
    w, s, e, n = bounds
    width, height = int(round((e - w) / pixel_deg)), int(round((n - s) / pixel_deg))
    transform = transform_from_bounds(w, s, e, n, width=width, height=height)
    codes = np.zeros((height, width), dtype="uint8")
    for url in _worldcover_tiles(bounds):
        try:
            src = rasterio.open(url)
        except RasterioIOError:
            codes[(codes == 0) & _cells_in_tile(url, codes.shape, transform)] = WATER_CODE
            continue
        with src:
            win = _window(src, bounds)
            if win is None:
                continue
            # Decimated read: GDAL serves this from the file's overviews.
            out_h = max(1, int(round(win.height * src.res[1] / pixel_deg)))
            out_w = max(1, int(round(win.width * src.res[0] / pixel_deg)))
            raw = src.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.mode)
            scaled = src.window_transform(win) * Affine.scale(win.width / out_w, win.height / out_h)
            dst = np.zeros_like(codes)
            reproject(raw, dst, src_transform=scaled, src_crs=CRS, dst_transform=transform,
                      dst_crs=CRS, src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
            codes[codes == 0] = dst[codes == 0]
    return codes == WATER_CODE, transform


def _drop_small_water(water, dy_km, dx_km, min_km2):
    labels, n = ndimage.label(water)
    if n == 0:
        return water
    sizes = ndimage.sum(water, labels, index=np.arange(1, n + 1)) * dy_km * dx_km
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_km2
    # Water touching the window edge is part of something bigger, so keep it.
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep[edge[edge > 0]] = True
    return keep[labels]


def _sector_kernels(dy_km, dx_km):
    r_px = int(np.ceil(SECTOR_RADIUS_KM / min(dy_km, dx_km)))
    di, dj = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    north, east = -di * dy_km, dj * dx_km
    dist = np.hypot(north, east)
    bearing = (np.rad2deg(np.arctan2(east, north)) + 360) % 360
    sector = np.floor(((bearing + 360 / N_SECTORS / 2) % 360) / (360 / N_SECTORS)).astype(int)
    kernels = []
    for k in range(N_SECTORS):
        ker = ((dist > 0) & (dist <= SECTOR_RADIUS_KM) & (sector == k)).astype("float32")
        kernels.append(ker / ker.sum())
    return kernels


def sea_features(bounds, pixel_deg, min_water_km2):
    """Sea distance and sea shares as coarse fields over a window.

    Returns fields plus the coarse transform, so they can be sampled at any
    point inside the window:
      dist_sea_km      distance to the nearest large water body
      sea_frac_10km    share of water within 10 km
      sea_frac_25km    share of water in a 0.25 degree box (about one ERA5 cell)
      sea_sector_k     share of water within 10 km in compass sector k
                       (k = 0 is north, 16 sectors of 22.5 degrees)
    """
    water, transform = _water_mask(bounds, pixel_deg)
    lat_mid = (bounds[1] + bounds[3]) / 2
    dy_km = pixel_deg * 110.57
    dx_km = pixel_deg * 111.32 * np.cos(np.deg2rad(lat_mid))
    water = _drop_small_water(water, dy_km, dx_km, min_water_km2)

    if water.any():
        dist = ndimage.distance_transform_edt(~water, sampling=(dy_km, dx_km))
    else:
        dist = np.full(water.shape, np.inf)
    # Beyond the search window we only know "far inland", so cap it there.
    fields = {"dist_sea_km": np.minimum(dist, MAX_SEA_KM)}
    w = water.astype("float32")
    sectors = []
    for k, ker in enumerate(_sector_kernels(dy_km, dx_km)):
        # Correlation (flipped kernel) so each sector looks outward from the cell.
        field = signal.fftconvolve(w, ker[::-1, ::-1], mode="same").clip(0, 1)
        fields[f"sea_sector_{k:02d}"] = field
        sectors.append(field)
    fields["sea_frac_10km"] = np.mean(sectors, axis=0)
    box = max(1, int(round(0.25 / pixel_deg)))
    fields["sea_frac_25km"] = ndimage.uniform_filter(w, size=box, mode="nearest")
    return fields, transform


def sample(fields, transform, lats, lons):
    """Read coarse fields at points (nearest coarse pixel)."""
    rows, cols = rasterio.transform.rowcol(transform, np.asarray(lons), np.asarray(lats))
    rows, cols = np.asarray(rows), np.asarray(cols)
    shape = next(iter(fields.values())).shape
    rows, cols = rows.clip(0, shape[0] - 1), cols.clip(0, shape[1] - 1)
    return {k: v[rows, cols] for k, v in fields.items()}


# --------------------------------------------------------------- public API

def snap_to_land(stations: pd.DataFrame, max_km: float = 5.0, min_inland_km: float = 0.2,
                 pixel_deg: float = 0.0005, half_deg: float = 0.06) -> pd.DataFrame:
    """Move stations whose listed position is on water to the nearest land.

    Meteostat positions are often rounded to 0.01 to 0.05 degrees, which puts
    some coastal airports out at sea, and their features would then describe
    open water. A station more than ``max_km`` from land is left where it is;
    the table step drops it because its cell is still water.
    """
    out = stations.copy()
    out["listed_lat"], out["listed_lon"], out["moved_km"] = out["lat"], out["lon"], 0.0
    for i, st in out.iterrows():
        bounds = (st.lon - half_deg, st.lat - half_deg, st.lon + half_deg, st.lat + half_deg)
        water, t = _water_mask(bounds, pixel_deg)
        col, row = ~t * (st.lon, st.lat)
        r, c = int(row), int(col)
        if not water[r, c]:
            continue
        dy = pixel_deg * 111.32
        dx = dy * np.cos(np.deg2rad(st.lat))
        # Target land at least min_inland_km from water, so the station's own
        # 150 m cell is land rather than a strip of beach.
        depth = ndimage.distance_transform_edt(~water, sampling=(dy, dx))
        inland = depth >= min_inland_km
        dist, (ri, ci) = ndimage.distance_transform_edt(~inland, sampling=(dy, dx), return_indices=True)
        if dist[r, c] > max_km:
            continue
        lon, lat = t * (ci[r, c] + 0.5, ri[r, c] + 0.5)
        out.loc[i, ["lat", "lon", "moved_km"]] = [round(lat, 5), round(lon, 5), round(float(dist[r, c]), 2)]
        print(f"  moved {st['name']} {dist[r, c]:.2f} km from water to land", flush=True)
    return out


def station_features(stations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row of static features per station, read at the station's cell."""
    fc = cfg["features"]
    p = cfg["geohash_precision"]
    rows = []
    for st in stations.itertuples():
        lats, lons, bounds = geogrid.grid_around(st.lat, st.lon, fc["far_radius_cells"], p)
        fine = fine_grid_features(lats, lons, bounds, fc["near_radius_cells"], fc["far_radius_cells"])
        c = fc["far_radius_cells"]
        rec = {"station": st.station, "geohash": geogrid.encode(st.lat, st.lon, p)}
        rec |= {k: float(v[c, c]) for k, v in fine.items()}
        win = fc["coast_window_deg"]
        coarse, t = sea_features((st.lon - win, st.lat - win, st.lon + win, st.lat + win),
                                 fc["coast_pixel_deg"], fc["min_water_body_km2"])
        rec |= {k: float(v[0]) for k, v in sample(coarse, t, [st.lat], [st.lon]).items()}
        rows.append(rec)
        print(f"  features {len(rows)}/{len(stations)}: {st.name}")
    return pd.DataFrame(rows)


def map_features(cfg: dict) -> pd.DataFrame:
    """Static features for every geohash cell in the map area."""
    fc, m, p = cfg["features"], cfg["map"], cfg["geohash_precision"]
    dlat, dlon = geogrid.cell_size(p)
    pad = fc["far_radius_cells"]
    lats, lons, bounds = geogrid.aligned_grid(m["lat_min"] - pad * dlat, m["lat_max"] + pad * dlat,
                                              m["lon_min"] - pad * dlon, m["lon_max"] + pad * dlon, p)
    fine = fine_grid_features(lats, lons, bounds, fc["near_radius_cells"], fc["far_radius_cells"])
    inner = (slice(pad, -pad), slice(pad, -pad))
    lat2d, lon2d = np.meshgrid(lats[inner[0]], lons[inner[1]], indexing="ij")
    df = pd.DataFrame({"lat": lat2d.ravel(), "lon": lon2d.ravel()})
    for k, v in fine.items():
        df[k] = v[inner].ravel()
    win = fc["coast_window_deg"]
    coarse, t = sea_features((m["lon_min"] - win, m["lat_min"] - win, m["lon_max"] + win, m["lat_max"] + win),
                             fc["coast_pixel_deg"], fc["min_water_body_km2"])
    for k, v in sample(coarse, t, df["lat"], df["lon"]).items():
        df[k] = v
    df["geohash"] = [geogrid.encode(a, b, p) for a, b in zip(df["lat"], df["lon"])]
    df.attrs["grid_shape"] = lat2d.shape
    return df
