"""Small checks of rules that decide which data is used."""

import numpy as np
import pandas as pd

from gulfheat import dataset, evaluate
from gulfheat.config import PATHS


def test_drop_untrusted_drops_height_mismatch_and_water(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "PATHS", type(PATHS)(root=tmp_path).ensure())
    meta = pd.DataFrame({"station": ["a", "b", "c"], "name": ["A", "B", "C"], "country": ["SA"] * 3,
                         "lat": [26.0] * 3, "lon": [50.0] * 3, "elev_station": [10, 3, 20]})
    static = pd.DataFrame({"station": ["a", "b", "c"], "elev": [15.0, 1413.0, 0.0],
                           "water_near": [0.0, 0.0, 1.0]})
    kept = dataset.drop_untrusted(meta, static)
    assert kept["station"].tolist() == ["a"]
    dropped = pd.read_csv(tmp_path / "data" / "processed" / "dropped_stations.csv")
    assert set(dropped["station"]) == {"b", "c"}


def test_station_folds_keep_whole_stations_and_use_every_fold():
    df = pd.DataFrame({"station": [f"s{i}" for i in range(14) for _ in range(3)],
                       "country": ["SA", "IR"] * 21})
    folds = evaluate.station_folds(df, k=7, seed=0)
    assert len(folds) == 14
    assert sorted(set(folds.values())) == list(range(7))


def test_hourly_features_exclude_fixed_place_features():
    assert not set(dataset.HOURLY_FEATURES) & {"elev", "dist_sea_km", "built_far"}
    assert set(dataset.HOURLY_FEATURES) | set(dataset.PLACE_FEATURES) == set(dataset.FEATURES)


def test_upwind_sea_picks_the_sector_the_wind_comes_from():
    n = dataset.N_SECTORS
    row = {f"sea_sector_{k:02d}": float(k == 4) for k in range(n)}   # sector 4 = east
    df = pd.DataFrame([row | {"wind_from": 90.0}, row | {"wind_from": 270.0}])
    df = df.assign(era5_t2m=30.0, era5_d2m=10.0, elev=5.0, z_era5=5.0, era5_wind=4.0,
                   time=pd.Timestamp("2024-06-19 11:00"), utc_offset=3)
    out = dataset.add_features(df)
    assert out["upwind_sea"].tolist() == [1.0, 0.0]
    assert np.isclose(out["local_hour"].iloc[0], 14)


def test_height_constraint_holds_even_when_data_says_otherwise():
    from gulfheat import models
    rng = np.random.default_rng(0)
    feats = dataset.FEATURE_SETS["lightgbm_cell"]
    df = pd.DataFrame(rng.normal(size=(3000, len(feats))), columns=feats)
    df["elev_minus_era5"] = rng.uniform(-300, 300, len(df))
    # Training data where higher sites are warmer: the constraint must not follow it.
    df["residual"] = 0.01 * df["elev_minus_era5"] + rng.normal(0, 0.1, len(df))
    m = models.make("lightgbm_cell", {"lightgbm_cell": {"n_estimators": 50, "min_child_samples": 20}}).fit(df)
    probe = pd.concat([df.iloc[[0]]] * 61, ignore_index=True)
    probe["elev_minus_era5"] = np.linspace(-300, 300, 61)
    assert np.all(np.diff(m.predict(probe)) <= 1e-9)
    # The first design's reference model has no constraint and follows the data.
    ref = models.make("lightgbm_first_design", {"lightgbm_first_design": {"n_estimators": 50}})
    assert "monotone_constraints" not in ref.params


def test_sea_share_blends_sea_grid_points_like_era5_values():
    import xarray as xr
    from gulfheat import era5
    times = pd.date_range("2024-01-01", periods=2, freq="h")
    lat, lon = [26.0, 26.25], [50.0, 50.25]
    shape = (len(times), 2, 2)
    ds = xr.Dataset({v: (("time", "latitude", "longitude"), np.ones(shape, "float32"))
                     for v in ["t2m", "d2m", "ssrd", "tcc", "u10", "v10"]}
                    | {"z_era5": (("latitude", "longitude"), np.array([[0.0, 0.0], [12.0, 30.0]]))},
                    coords={"time": times, "latitude": lat, "longitude": lon})
    pts = era5.at_points(ds, [26.0, 26.125, 26.25], [50.1, 50.1, 50.1])
    assert np.allclose(pts["sea_point"].values, [1.0, 0.5, 0.0])


def test_test_stations_need_training_rows_and_map_uses_constrained_models():
    from gulfheat import models
    df = pd.DataFrame({"station": ["a", "a", "b"], "year": [2024, 2025, 2025]})
    train, test = evaluate._split(df, {"train_years": [2024], "test_years": [2025]})
    assert test["station"].tolist() == ["a"]
    assert not {"era5_raw", "lightgbm_first_design", "mlp"} & set(models.MAP_CANDIDATES)
