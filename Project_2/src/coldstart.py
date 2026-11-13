"""Cold-start model: vehicles with NO reading before the cutoff.

Only static attributes and the target date are known. Training samples mimic this: for many
cutoffs c, vehicles whose first reading is after c, targets = their readings in (c, c+horizon].
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import DOY, MONTH, YEAR
from .series import KEY

COLD_NUM = ["h", "tgt_month", "tgt_doy", "age_t", "power_ps", "cubic_capacity_liter",
            "model_year", "acriss_code_anonymized"]
COLD_CAT = ["vehicle_level", "vehicle_type", "driven_wheels"]
COLD_FEATS = COLD_NUM + COLD_CAT


def cold_features(t: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Feature frame for rows with columns KEY, day (target day) + static attributes."""
    d = t.copy()
    dt = d.day.to_numpy()
    d["h"] = dt - cutoff
    d["tgt_month"], d["tgt_doy"] = MONTH[dt], DOY[dt]
    d["age_t"] = YEAR[dt] - d.model_year
    return d


def cold_samples(all_: pd.DataFrame, vehicles, cutoffs, horizon=150, max_day=None,
                 max_targets=8, rng=None) -> pd.DataFrame:
    """Rows (clean readings) of vehicles whose first reading lies after each cutoff."""
    a = all_[all_[KEY].isin(vehicles) & all_.ok]
    first = a.groupby(KEY).day.min()
    parts = []
    for c in cutoffs:
        v = first.index[first > c]
        hi = c + horizon if max_day is None else min(c + horizon, max_day)
        t = a[a[KEY].isin(v) & (a.day > c) & (a.day <= hi)]
        if max_targets:
            t = t.assign(_r=rng.random(len(t)), _n=t.groupby(KEY)[KEY].transform("size"))
            t = t.sort_values("_r").groupby(KEY).head(max_targets)
            t = t.assign(w_tgt=t._n / np.minimum(t._n, max_targets)).drop(columns=["_r", "_n"])
        else:
            t = t.assign(w_tgt=1.0)
        if len(t):
            parts.append(cold_features(t, c))
    return pd.concat(parts, ignore_index=True)


def fit_cold(train: pd.DataFrame, seed=0, n_estimators=300) -> lgb.LGBMRegressor:
    """L1 regression of km at the target date on static attributes + calendar."""
    m = lgb.LGBMRegressor(objective="l1", n_estimators=n_estimators, learning_rate=0.05,
                          num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, n_jobs=16, verbose=-1, random_state=seed)
    return m.fit(train[COLD_FEATS], train.km, sample_weight=train.w_tgt)
