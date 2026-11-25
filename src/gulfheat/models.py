"""Models that predict ERA5's error at a site.

All models share one interface: ``fit(train_df)`` and ``predict(df)``, which
returns the predicted residual (observed minus ERA5) in degrees C. The first
three are baselines a learned model has to beat.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset import FEATURE_SETS, FEATURES, MONOTONE, REFERENCE_SETS

torch.set_num_threads(1)  # see MLP.fit: avoids an OpenMP clash with LightGBM on macOS

LAPSE_RATE = -0.0065  # degrees C per metre, the standard atmosphere


class RawERA5:
    """No correction: use ERA5 as it is."""
    name = "era5_raw"

    def fit(self, df):
        return self

    def predict(self, df):
        return np.zeros(len(df))


class LapseRate:
    """Correct only for the height difference between the site and the ERA5 cell."""
    name = "lapse_rate"

    def fit(self, df):
        return self

    def predict(self, df):
        return LAPSE_RATE * df["elev_minus_era5"].to_numpy()


class HourlyBias:
    """ERA5's average error at each local hour, learned across training stations."""
    name = "hourly_bias"

    def fit(self, df):
        self.bias = df.groupby("local_hour")["residual"].mean()
        return self

    def predict(self, df):
        return df["local_hour"].map(self.bias).fillna(0).to_numpy()


class GBM:
    """LightGBM on a chosen feature set (all features by default).

    ``monotone`` maps feature names to -1 or +1: the prediction may only fall
    (or rise) as that feature grows, whatever the other features are.
    """

    def __init__(self, features=None, seed=0, name="lightgbm", monotone=None, **params):
        self.name = name
        self.features = list(features or FEATURES)
        self.params = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31,
                       "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
                       "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1,
                       "random_state": seed} | params
        if monotone and any(f in monotone for f in self.features):
            self.params |= {"monotone_constraints": [monotone.get(f, 0) for f in self.features],
                            # "intermediate" is less restrictive than the basic method
                            # while still guaranteeing the constraint.
                            "monotone_constraints_method": "intermediate"}

    def fit(self, df):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(df[self.features], df["residual"])
        return self

    def predict(self, df):
        return self.model.predict(df[self.features])

    def importance(self) -> pd.Series:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=self.features).sort_values(ascending=False)


class _Net(nn.Module):
    def __init__(self, n_in, hidden=128, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class MLP:
    """A small PyTorch network on the same features.

    Uses a Huber loss, which treats large errors linearly, so a few bad
    station readings pull on it less than they would with squared error.
    Training stops when the loss on a set of held-back stations stops
    improving, so it never sees the test stations while choosing when to stop.
    """
    name = "mlp"

    def __init__(self, features=None, seed=0, epochs=40, batch=8192, lr=3e-3, patience=5):
        self.features = list(features or FEATURES)
        self.seed, self.epochs, self.batch, self.lr, self.patience = seed, epochs, batch, lr, patience
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _x(self, df):
        x = (df[self.features].to_numpy("float32") - self.mean) / self.std
        return torch.from_numpy(x.astype("float32"))

    def fit(self, df):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        stations = df["station"].unique()
        val_st = rng.choice(stations, size=max(1, len(stations) // 6), replace=False)
        tr, va = df[~df["station"].isin(val_st)], df[df["station"].isin(val_st)]

        self.mean = tr[self.features].mean().to_numpy("float32")
        self.std = tr[self.features].std().replace(0, 1).to_numpy("float32")
        # Everything lives on the device and batches are indexed there. CPU
        # tensor ops would start torch's own OpenMP threads, and on macOS two
        # OpenMP runtimes in one process (LightGBM's and torch's) can deadlock.
        dev = self.device
        xtr, ytr = self._x(tr).to(dev), torch.tensor(tr["residual"].to_numpy("float32")).to(dev)
        xva, yva = self._x(va).to(dev), torch.tensor(va["residual"].to_numpy("float32")).to(dev)
        gen = torch.Generator(device=dev).manual_seed(self.seed)

        self.net = _Net(len(self.features)).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.HuberLoss(delta=1.0)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(xtr), device=dev, generator=gen)
            for i in range(0, len(order), self.batch):
                idx = order[i:i + self.batch]
                xb, yb = xtr[idx], ytr[idx]
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(xva), yva).item()
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, df):
        self.net.eval()
        with torch.no_grad():
            out = [self.net(chunk.to(self.device)).cpu().numpy()
                   for chunk in torch.split(self._x(df), 65536)]
        return np.concatenate(out)


def make(name: str, gbm_params: dict | None = None, seed: int = 0):
    """Build a model by the name used in result tables.

    The design's LightGBM models carry the height constraint; the first
    design's reference model does not, exactly as it was first tested.
    """
    params = (gbm_params or {}).get(name, {})
    if name in FEATURE_SETS:
        return GBM(features=FEATURE_SETS[name], seed=seed, name=name, monotone=MONOTONE, **params)
    if name in REFERENCE_SETS:
        return GBM(features=REFERENCE_SETS[name], seed=seed, name=name, **params)
    return {"era5_raw": RawERA5, "lapse_rate": LapseRate, "hourly_bias": HourlyBias,
            "mlp": lambda: MLP(seed=seed)}[name]()


MODEL_NAMES = ["era5_raw", "lapse_rate", "hourly_bias", *FEATURE_SETS, *REFERENCE_SETS, "mlp"]
# Models allowed to draw the map: ones that cannot make higher ground warmer.
# Raw ERA5 is the reference, the first design is what is being replaced, and
# the MLP has no height constraint.
MAP_CANDIDATES = ["lapse_rate", "hourly_bias", *FEATURE_SETS]
