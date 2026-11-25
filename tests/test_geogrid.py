import numpy as np

from gulfheat import geogrid


def test_encode_matches_reference():
    # Reference value from the original geohash description.
    assert geogrid.encode(57.64911, 10.40744, 11) == "u4pruydqqvj"


def test_cell_size_precision_7():
    dlat, dlon = geogrid.cell_size(7)
    assert np.isclose(dlat, 180 / 2**17)
    assert np.isclose(dlon, 360 / 2**18)


def test_grid_cells_are_geohash_cells():
    lats, lons, _ = geogrid.aligned_grid(26.25, 26.30, 50.10, 50.15, 7)
    # Every cell centre must encode to a distinct geohash, and neighbours must differ.
    hashes = {geogrid.encode(a, b, 7) for a in lats for b in lons}
    assert len(hashes) == len(lats) * len(lons)


def test_grid_around_is_centred():
    lats, lons, _ = geogrid.grid_around(26.2667, 50.15, 5, 7)
    assert len(lats) == len(lons) == 11
    assert geogrid.encode(lats[5], lons[5], 7) == geogrid.encode(26.2667, 50.15, 7)
