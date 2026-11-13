"""Evaluation harness: simulated reporting-lag pairs, splits and metrics.

Setup: for every observation j (the "truth") and every lag L we pretend the last known reading
is the latest observation i with date_i <= date_j - L. A model must predict km_j from what is
known at i. All pair features use observations up to and including i only (no leakage).
"""
import numpy as np
import pandas as pd

from .data import KEY

LAGS = (1, 7, 14, 28)
RATE_WINDOW = 90  # days for the "recent rate" feature


def make_pairs(tl: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    """One row per (target observation j, lag): last-known state at i and the true km at j."""
    rows = []
    for vid, g in tl.groupby(KEY, sort=False):
        day, km = g.day.to_numpy(), g.km.to_numpy()
        seg = g[["vehicle_level", "vehicle_type"]].iloc[0].tolist()
        for j in range(1, len(g)):
            for lag in lags:
                i = np.searchsorted(day, day[j] - lag, side="right") - 1
                if i < 0:
                    continue
                span = day[i] - day[0]
                cum_rate = (km[i] - km[0]) / span if span > 0 else np.nan
                k = np.searchsorted(day, day[i] - RATE_WINDOW, side="left")
                span90 = day[i] - day[k]
                rate90 = (km[i] - km[k]) / span90 if span90 > 0 else np.nan
                rows.append((vid, lag, day[i], day[j], day[j] - day[i], km[i], km[j],
                             cum_rate, rate90, i, seg[0], seg[1]))
    cols = ["vid", "lag", "day0", "day_t", "gap", "km0", "km_true", "cum_rate", "rate90",
            "n_hist", "vehicle_level", "vehicle_type"]
    return pd.DataFrame(rows, columns=cols)


def metrics(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """MAE, median AE, WAPE (sum abs err / sum abs true delta) and signed bias, all in km."""
    err = df[pred_col] - df.km_true
    delta = (df.km_true - df.km0).abs().sum()
    return pd.Series({
        "n": len(df),
        "MAE": err.abs().mean(),
        "MedAE": err.abs().median(),
        "WAPE": err.abs().sum() / delta if delta > 0 else np.nan,
        "bias": err.mean(),
    })


def metrics_table(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Rows: model x lag (plus 'all'); columns: n, MAE, MedAE, WAPE, bias."""
    out = []
    for c in pred_cols:
        for lag, sub in [("all", df)] + [(lg, s) for lg, s in df.groupby("lag")]:
            out.append(metrics(sub, c).rename((c, lag)))
    return pd.DataFrame(out).rename_axis(["model", "lag"])


def vehicle_split(vehicles, test_frac=0.3, seed=0):
    """Random split of vehicle ids into (train, test) sets."""
    rng = np.random.default_rng(seed)
    v = np.array(sorted(vehicles))
    rng.shuffle(v)
    n_test = max(1, int(round(len(v) * test_frac)))
    return set(v[n_test:]), set(v[:n_test])


def time_cutoff(tl: pd.DataFrame, frac_test=0.2) -> int:
    """Day number such that ~frac_test of all observations happen on/after it."""
    return int(tl.day.quantile(1 - frac_test))
