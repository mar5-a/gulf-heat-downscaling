"""Geohash cells and a latitude/longitude grid that lines up with them.

A geohash splits longitude and latitude in half again and again, alternating,
and writes the bits in base 32. At precision 7 there are 18 longitude bits and
17 latitude bits, so every cell is 360/2**18 by 180/2**17 degrees: about
0.00137 degrees, or roughly 150 m at Gulf latitudes. Because the cells form a
regular grid, raster data can be averaged straight onto them.
"""

from __future__ import annotations

import numpy as np

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def cell_size(precision: int) -> tuple[float, float]:
    """Return (dlat, dlon) in degrees for one geohash cell."""
    bits = 5 * precision
    lon_bits = (bits + 1) // 2
    lat_bits = bits // 2
    return 180.0 / 2**lat_bits, 360.0 / 2**lon_bits


def encode(lat: float, lon: float, precision: int = 7) -> str:
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    chars, bit, ch, even = [], 0, 0, True
    while len(chars) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                ch |= 1 << (4 - bit)
                lon_lo = mid
            else:
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch |= 1 << (4 - bit)
                lat_lo = mid
            else:
                lat_hi = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            chars.append(BASE32[ch])
            bit, ch = 0, 0
    return "".join(chars)


def aligned_grid(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, precision: int = 7
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Cell-centre latitudes (north to south) and longitudes covering a box.

    The box is widened to the nearest geohash cell edges, so each grid cell is
    exactly one geohash cell. Also returns the snapped bounds
    (west, south, east, north) for building a raster transform.
    """
    dlat, dlon = cell_size(precision)
    south = -90 + np.floor((lat_min + 90) / dlat) * dlat
    north = -90 + np.ceil((lat_max + 90) / dlat) * dlat
    west = -180 + np.floor((lon_min + 180) / dlon) * dlon
    east = -180 + np.ceil((lon_max + 180) / dlon) * dlon
    n_lat = int(round((north - south) / dlat))
    n_lon = int(round((east - west) / dlon))
    lats = north - dlat * (np.arange(n_lat) + 0.5)
    lons = west + dlon * (np.arange(n_lon) + 0.5)
    return lats, lons, (west, south, east, north)


def grid_around(lat: float, lon: float, half_cells: int, precision: int = 7):
    """A square grid of (2 * half_cells + 1) cells centred on the cell holding (lat, lon)."""
    dlat, dlon = cell_size(precision)
    c_lat = -90 + (np.floor((lat + 90) / dlat) + 0.5) * dlat
    c_lon = -180 + (np.floor((lon + 180) / dlon) + 0.5) * dlon
    return aligned_grid(
        c_lat - half_cells * dlat,
        c_lat + half_cells * dlat,
        c_lon - half_cells * dlon,
        c_lon + half_cells * dlon,
        precision,
    )
