"""Street-level temperature map for Dammam, Khobar and Dhahran.

ERA5 is interpolated to every 150 m geohash cell, then the correction from the
model chosen by cross-validation on held-out stations in the training years
(and trained on those years) is added. Cells that are mostly water are left
blank. This map extrapolates from airport stations into city centres, so it is
a model estimate, not a measurement; the README says how far to trust it.

The first design's model is run on the same cells as a check: it used absolute
height and drew the Dhahran dome's height contours onto the map.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import era5, models as M, static_features
from .config import PATHS
from .dataset import REFERENCE_SETS, add_features, era5_series, load_table

UTC_OFFSET_MAP = 3  # Saudi Arabia


def map_static(cfg) -> pd.DataFrame:
    path = PATHS.processed / "map_static.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.attrs["grid_shape"] = tuple(json.loads((PATHS.processed / "map_shape.json").read_text()))
        return df
    df = static_features.map_features(cfg)
    df.to_parquet(path)
    (PATHS.processed / "map_shape.json").write_text(json.dumps(list(df.attrs["grid_shape"])))
    return df


def best_model(cfg, name=None):
    """The model chosen in evaluate by cross-validation on held-out stations, or another by name."""
    res = json.loads((PATHS.reports / "metrics.json").read_text())
    return M.make(name or res["map_model"], res["gbm_params"], cfg["seed"])


def make_map(cfg: dict) -> pd.DataFrame:
    static = map_static(cfg)
    shape = static.attrs["grid_shape"]
    targets = pd.to_datetime(cfg["map"]["times_utc"])
    # Three earlier hours are needed for the 1 h and 3 h temperature changes.
    hours = sorted({t - pd.Timedelta(hours=h) for t in targets for h in range(4)})
    cube = era5.open_cube()
    e = era5_series(cube, static["lat"], static["lon"], static.index.to_numpy(), times=hours)
    e = e[e["time"].isin(targets)].rename(columns={"site": "cell"})
    df = e.merge(static, left_on="cell", right_index=True)
    df["utc_offset"] = UTC_OFFSET_MAP
    df = add_features(df)

    table = load_table()
    train = table[table["year"].isin(cfg["train_years"])]
    df["correction"] = best_model(cfg).fit(train).predict(df)
    df["correction_first_design"] = best_model(cfg, next(iter(REFERENCE_SETS))).fit(train).predict(df)
    df["t2m_corrected"] = df["era5_t2m"] + df["correction"]
    water = df["water_near"] > 0.5
    df.loc[water, ["correction", "correction_first_design", "t2m_corrected"]] = np.nan
    out = df[["geohash", "lat", "lon", "time", "elev", "era5_t2m", "correction", "correction_first_design",
              "t2m_corrected", "built_near", "tree_near", "dist_sea_km"]]
    out.to_parquet(PATHS.processed / "map_predictions.parquet")
    by_height(out[~water.to_numpy()]).to_csv(PATHS.reports / "tables" / "map_by_height.csv")
    plot(out, shape, cfg)
    plot_designs(out, shape, cfg)
    return out


def by_height(df: pd.DataFrame) -> pd.DataFrame:
    """Mean correction on land cells by terrain height, for both designs.

    No station sits on the Dhahran dome, so this is the only check of whether
    the map follows height contours. Cells within 3 km of the sea are left out,
    because low ground is mostly coast and the sea breeze would hide the effect.
    """
    d = df[df["dist_sea_km"] >= 3].copy()
    d["height"] = pd.cut(d["elev"], [-50, 10, 25, 1000], labels=["below 10 m", "10 to 25 m", "above 25 m"])
    local = (d["time"] + pd.Timedelta(hours=UTC_OFFSET_MAP)).dt.strftime("%H:%M local")
    g = d.groupby([local.rename("time"), "height"], observed=True)
    return g[["correction", "correction_first_design"]].mean().round(2).assign(cells=g.size())


def plot_designs(df: pd.DataFrame, shape, cfg):
    """The 2 pm correction from the first design next to the current one."""
    t = sorted(df["time"].unique())[0]
    d = df[df["time"] == t].sort_values(["lat", "lon"], ascending=[False, True])
    ext = [d["lon"].min(), d["lon"].max(), d["lat"].min(), d["lat"].max()]
    cols = [("correction_first_design", "First design (uses absolute height)"),
            ("correction", "Current design (height can only cool)")]
    lim = np.nanpercentile(np.abs(d[[c for c, _ in cols]].to_numpy()), 98)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    local = (pd.Timestamp(t) + pd.Timedelta(hours=UTC_OFFSET_MAP)).strftime("%d %b %Y, %H:%M local")
    for ax, (col, title) in zip(axes, cols):
        im = ax.imshow(d[col].to_numpy().reshape(shape), extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       origin="upper")
        ax.set_title(f"{title}\ncorrection added to ERA5, {local}", fontsize=10)
        ax.plot(50.15, 26.2667, marker="^", color="white", mec="black", ms=7)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="°C")
    fig.savefig(PATHS.figures / "map_first_vs_current_design.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot(df: pd.DataFrame, shape, cfg):
    times = sorted(df["time"].unique())
    fig, axes = plt.subplots(2, len(times), figsize=(5.2 * len(times), 9.4))
    axes = np.atleast_2d(axes).reshape(2, len(times))
    m = cfg["map"]
    extent = [m["lon_min"], m["lon_max"], m["lat_min"], m["lat_max"]]
    for j, t in enumerate(times):
        d = df[df["time"] == t].sort_values(["lat", "lon"], ascending=[False, True])
        local = (pd.Timestamp(t) + pd.Timedelta(hours=UTC_OFFSET_MAP)).strftime("%d %b %Y, %H:%M local")
        temp = d["t2m_corrected"].to_numpy().reshape(shape)
        corr = d["correction"].to_numpy().reshape(shape)
        ext = [d["lon"].min(), d["lon"].max(), d["lat"].min(), d["lat"].max()]
        im = axes[0, j].imshow(temp, extent=ext, cmap="inferno", origin="upper")
        axes[0, j].set_title(f"Corrected temperature\n{local}", fontsize=10)
        fig.colorbar(im, ax=axes[0, j], shrink=0.8, label="°C")
        lim = np.nanpercentile(np.abs(corr), 98)
        im2 = axes[1, j].imshow(corr, extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper")
        axes[1, j].set_title("Correction added to ERA5", fontsize=10)
        fig.colorbar(im2, ax=axes[1, j], shrink=0.8, label="°C")
        for ax in axes[:, j]:
            ax.plot(50.15, 26.2667, marker="^", color="white", mec="black", ms=7)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.tick_params(labelsize=8)
    fig.suptitle("Dammam, Khobar and Dhahran on 150 m geohash cells (triangle: Dhahran station)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PATHS.figures / "street_level_map.png", dpi=160)
    plt.close(fig)
