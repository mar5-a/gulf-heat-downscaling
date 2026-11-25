"""Evaluation: two test settings, breakdowns, a small tuning study and ablations.

Settings (every score is on held-out test-year readings, 2025 in the config):
  seen_stations    train on the training years at all stations, test on the
                   test year at the same stations. Tests "a later year, same
                   places".
  unseen_stations  stations are split into groups; train at the other
                   groups, test at the held-out group in the test year.
                   Tests "a later year, a place the model never saw", which
                   is what a street-level map needs.

Only stations with training-year readings are scored, so "seen" really means
seen. Tuning and the choice of model for the map only use the training years,
with stations held out by group, so test scores never choose anything.
"""

from __future__ import annotations

import json
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import models as M
from .config import PATHS
from .dataset import FEATURE_GROUPS, FEATURE_SETS, FEATURES, MONOTONE, REFERENCE_SETS

LABELS = {"era5_raw": "ERA5 as is", "lapse_rate": "Height correction",
          "hourly_bias": "Average hourly bias", "lightgbm_hourly": "LightGBM, hourly features",
          "lightgbm_cell": "LightGBM, + ERA5 cell", "lightgbm_sea": "LightGBM, + ERA5 cell and sea",
          "lightgbm": "LightGBM, all features", "lightgbm_first_design": "LightGBM, first design",
          "mlp": "PyTorch MLP, all features"}


def metrics(y, yhat) -> dict:
    err = np.asarray(yhat) - np.asarray(y)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)), "n": int(len(err))}


def _split(df, cfg):
    train = df[df["year"].isin(cfg["train_years"])]
    test = df[df["year"].isin(cfg["test_years"]) & df["station"].isin(train["station"])]
    return train, test


def _years(cfg, key):
    return "-".join(str(y) for y in sorted({min(cfg[key]), max(cfg[key])}))


def station_folds(df, k, seed):
    """Assign each station to one of k groups, mixing countries across groups."""
    st = df.groupby("station")["country"].first().sample(frac=1, random_state=seed).sort_values(kind="stable")
    return {s: i % k for i, s in enumerate(st.index)}


def in_box(df, box):
    return (df["lat"].between(box["lat_min"], box["lat_max"])
            & df["lon"].between(box["lon_min"], box["lon_max"]))


# ------------------------------------------------------------------ tuning

def tune_gbm(train: pd.DataFrame, cfg: dict, label) -> tuple[dict, pd.DataFrame]:
    """Grid search on the training years only, scored on held-out stations."""
    grid = {"num_leaves": [15, 31, 63], "min_child_samples": [100, 2000]}
    folds = list(GroupKFold(n_splits=5).split(train, groups=train["station"]))
    rows = []
    for values in product(*grid.values()):
        params = dict(zip(grid, values)) | {"learning_rate": 0.05, "n_estimators": 300}
        maes = []
        for tr_idx, va_idx in folds:
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            m = M.make(label, {label: params}, cfg["seed"]).fit(tr)
            maes.append(np.mean(np.abs(m.predict(va) - va["residual"])))
        rows.append({"model": label} | params | {"cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))})
        print(f"  tune {label} {params} -> {rows[-1]['cv_mae']:.3f}", flush=True)
    table = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    best = {k: table.iloc[0][k] for k in ["num_leaves", "min_child_samples", "learning_rate", "n_estimators"]}
    best = {k: (float(v) if k == "learning_rate" else int(v)) for k, v in best.items()}
    return best, table


def cv_scores(train: pd.DataFrame, cfg: dict, params: dict) -> pd.DataFrame:
    """Every model's error on held-out stations within the training years (5 station groups).

    This is how the map's model is chosen. GBM settings were tuned on the same
    folds, which flatters them slightly against the baselines.
    """
    folds = list(GroupKFold(n_splits=5).split(train, groups=train["station"]))
    rows = []
    for name in M.MODEL_NAMES:
        err = []
        for tr_idx, va_idx in folds:
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            err.append(np.abs(M.make(name, params, cfg["seed"]).fit(tr).predict(va) - va["residual"]))
        rows.append({"model": name, "cv_mae_train": float(np.concatenate(err).mean())})
        print(f"  training-years cross-validation {name}: {rows[-1]['cv_mae_train']:.3f}", flush=True)
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------- experiments

def _models(params: dict, seed: int):
    return [M.make(name, params, seed) for name in M.MODEL_NAMES]


def run_settings(df: pd.DataFrame, cfg: dict, params: dict) -> pd.DataFrame:
    """Predictions for every model in both settings, on test-year rows."""
    train, test = _split(df, cfg)
    out = test[["station", "name", "country", "lat", "lon", "time", "local_hour", "season", "dist_sea_km",
                "built_far", "obs_t2m", "era5_t2m", "residual"]].copy()

    for model in _models(params, cfg["seed"]):
        out[f"seen__{model.name}"] = model.fit(train).predict(test)
        print(f"  seen stations: {model.name} done", flush=True)

    fold_of = station_folds(df, cfg["spatial_folds"], cfg["seed"])
    test_fold = test["station"].map(fold_of)
    train_fold = train["station"].map(fold_of)
    for model in _models(params, cfg["seed"]):
        pred = np.full(len(test), np.nan)
        for f in range(cfg["spatial_folds"]):
            held = (test_fold == f).to_numpy()
            if not held.any():
                continue
            fitted = model.fit(train[train_fold != f])
            pred[held] = fitted.predict(test[held])
        out[f"unseen__{model.name}"] = pred
        print(f"  unseen stations: {model.name} done", flush=True)
    return out


def summarise(pred: pd.DataFrame, mask=None, scope="all") -> pd.DataFrame:
    p = pred if mask is None else pred[mask]
    rows = []
    for col in [c for c in p.columns if "__" in c]:
        setting, model = col.split("__")
        rows.append({"scope": scope, "setting": setting, "model": model} | metrics(p["residual"], p[col]))
    t = pd.DataFrame(rows)
    raw = t[t["model"] == "era5_raw"].set_index("setting")["mae"]
    t["mae_cut_vs_era5"] = 1 - t["mae"] / t["setting"].map(raw)
    return t


def breakdown(pred: pd.DataFrame, by: str, setting: str = "unseen") -> pd.DataFrame:
    cols = [c for c in pred.columns if c.startswith(f"{setting}__")]
    g = pred.groupby(by)
    t = pd.DataFrame({c.split("__")[1]: g.apply(lambda d, c=c: np.mean(np.abs(d[c] - d["residual"])),
                                                include_groups=False) for c in cols})
    t["rows"] = g.size()
    return t


def ablation(df: pd.DataFrame, cfg: dict, params: dict) -> pd.DataFrame:
    """Unseen-station MAE of the all-features LightGBM when each group is left out."""
    train, test = _split(df, cfg)
    fold_of = station_folds(df, cfg["spatial_folds"], cfg["seed"])
    test_fold, train_fold = test["station"].map(fold_of), train["station"].map(fold_of)

    def score(features):
        err = []
        for f in range(cfg["spatial_folds"]):
            held = test_fold == f
            if not held.any():
                continue
            m = M.GBM(features=features, seed=cfg["seed"], monotone=MONOTONE, **params).fit(train[train_fold != f])
            err.append(np.abs(m.predict(test[held]) - test.loc[held, "residual"]))
        return float(np.concatenate(err).mean())

    rows = [{"left_out": "nothing", "mae": score(FEATURES)}]
    for group, cols in FEATURE_GROUPS.items():
        rows.append({"left_out": group, "mae": score([f for f in FEATURES if f not in cols])})
        print(f"  ablation without {group}: {rows[-1]['mae']:.3f}", flush=True)
    t = pd.DataFrame(rows)
    t["mae_increase"] = t["mae"] - t.loc[0, "mae"]
    return t


# ------------------------------------------------------------------ figures

def plot_hourly(by_hour: pd.DataFrame, path, title, models):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col in models:
        ax.plot(by_hour.index, by_hour[col], marker="o", ms=3, label=LABELS[col])
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Mean absolute error (°C)")
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stations(by_station: pd.DataFrame, names: pd.Series, model: str, path, year):
    t = by_station.sort_values("era5_raw")
    fig, ax = plt.subplots(figsize=(8, 0.2 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["era5_raw"], height=0.4, label="ERA5 as is", color="#B4B2A9")
    ax.barh(y - 0.2, t[model], height=0.4, label=LABELS[model], color="#185FA5")
    ax.set_yticks(y)
    ax.set_yticklabels([str(names.get(s, s))[:28] for s in t.index], fontsize=6)
    ax.set_xlabel(f"Mean absolute error (°C), {year}, station unseen in training")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------- main

def run(df: pd.DataFrame, cfg: dict) -> dict:
    PATHS.ensure()
    tables = PATHS.reports / "tables"
    tables.mkdir(exist_ok=True)
    train, test = _split(df, cfg)
    test_year = _years(cfg, "test_years")
    skipped = sorted(set(df.loc[df["year"].isin(cfg["test_years"]), "station"]) - set(test["station"]))
    print(f"  {test['station'].nunique()} test stations; {len(skipped)} with no training rows not scored", flush=True)

    params, tuning = {}, []
    for label in [*FEATURE_SETS, *REFERENCE_SETS]:
        best, t = tune_gbm(train, cfg, label)
        params[label] = best
        tuning.append(t)
    pd.concat(tuning).to_csv(tables / "tuning.csv", index=False)

    # Choose the map's model on the training years only, among models that
    # cannot make higher ground warmer.
    cv = cv_scores(train, cfg, params)
    cv.round(4).to_csv(tables / "cv_train.csv")
    chosen = cv.loc[M.MAP_CANDIDATES, "cv_mae_train"].idxmin()

    pred = run_settings(df, cfg, params)
    pred.to_parquet(PATHS.processed / f"predictions_{test_year}.parquet")
    gulf = in_box(pred, cfg["gulf_box"])
    pred["region"] = np.where(gulf, "Gulf coast box", "rest of region")
    summary = pd.concat([summarise(pred), summarise(pred, gulf, "gulf_box")])
    summary.to_csv(tables / "summary.csv", index=False)

    unseen = summary[(summary["scope"] == "all") & (summary["setting"] == "unseen")].set_index("model")
    unseen_gulf = summary[(summary["scope"] == "gulf_box") & (summary["setting"] == "unseen")].set_index("model")

    pred["coast"] = np.where(pred["dist_sea_km"] < 5, "within 5 km of sea", "5 km or more inland")
    frames = {"by_hour": breakdown(pred, "local_hour"), "by_country": breakdown(pred, "country"),
              "by_season": breakdown(pred, "season"), "by_coast": breakdown(pred, "coast"),
              "by_region": breakdown(pred, "region"), "by_station": breakdown(pred, "station"),
              "by_hour_gulf": breakdown(pred[gulf], "local_hour")}
    for name, t in frames.items():
        t.round(3).to_csv(tables / f"{name}.csv")
    abl = ablation(df, cfg, params["lightgbm"])
    abl.to_csv(tables / "ablation.csv", index=False)
    # Gain importance of each LightGBM model, trained on all training years.
    importance = pd.concat({name: M.make(name, params, cfg["seed"]).fit(train).importance()
                            for name in [*FEATURE_SETS, *REFERENCE_SETS]}, axis=1)
    importance.round(4).to_csv(tables / "importance.csv", index_label="feature")

    names = pred.groupby("station")["name"].first()
    first = next(iter(REFERENCE_SETS))
    shown = list(dict.fromkeys(["era5_raw", "lapse_rate", "hourly_bias", first, chosen, "lightgbm"]))
    plot_hourly(frames["by_hour"], PATHS.figures / "error_by_hour.png",
                f"Error by hour of day, {test_year}, stations the model never saw", shown)
    plot_hourly(frames["by_hour_gulf"], PATHS.figures / "error_by_hour_gulf.png",
                f"Gulf coast stations only, {test_year}, never seen in training", shown)
    plot_stations(frames["by_station"], names, chosen, PATHS.figures / "error_by_station.png", test_year)

    by_station = frames["by_station"]
    result = {
        "train_years": cfg["train_years"],
        "test_years": cfg["test_years"],
        "stations": int(pred["station"].nunique()),
        "test_rows": int(len(pred)),
        "test_stations_without_training_rows": [str(s) for s in skipped],
        "gbm_params": params,
        "map_model": chosen,
        "map_model_chosen_by": ("lowest error on held-out stations in the training years (5 station groups), "
                                "among models that cannot make higher ground warmer"),
        "cv_train": cv.round(3)["cv_mae_train"].to_dict(),
        "map_model_unseen_mae_test": round(float(unseen.loc[chosen, "mae"]), 3),
        "era5_unseen_mae_test": round(float(unseen.loc["era5_raw", "mae"]), 3),
        "lowest_unseen_mae_test_model": unseen.drop(index="era5_raw")["mae"].idxmin(),
        "first_design_vs_map_model_unseen": {
            scope: {m: round(float(t.loc[m, "mae"]), 3) for m in ["era5_raw", first, chosen]}
            for scope, t in [("all", unseen), ("gulf_box", unseen_gulf)]},
        "summary": summary.round(3).to_dict(orient="records"),
        "ablation": abl.round(3).to_dict(orient="records"),
        "top_features": {name: importance[name].sort_values(ascending=False).head(8).round(3).to_dict()
                         for name in importance},
        "stations_worse_after_correction": [
            str(names[s]) for s, r in by_station.iterrows() if r[chosen] > r["era5_raw"]],
    }
    (PATHS.reports / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
