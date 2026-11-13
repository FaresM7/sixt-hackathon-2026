"""Cutoff-based forecasting samples.

A sample = (vehicle, cutoff C, target day t > C). Everything known at C is the vehicle's readings
up to C (its last known reading may be older than C: reporting lag). Target = km at day t.
All features use readings with day <= C only (no leakage); the seasonal table is fitted on an
explicitly given period.
"""
import numpy as np
import pandas as pd

from .series import EPOCH, KEY, STATE_COLS, STATIC_COLS

WINDOWS = (7, 30, 90, 180, 365)
DAY_MAX = 10_500
_DAYS = EPOCH + pd.to_timedelta(np.arange(DAY_MAX + 1), unit="D")
MONTH = np.asarray(_DAYS.month)
DOY = np.asarray(_DAYS.dayofyear)
YEAR = np.asarray(_DAYS.year)


def day_of(date: str) -> int:
    """'YYYY-MM-DD' -> day number."""
    return int((pd.Timestamp(date) - EPOCH).days)


def seasonal_cumulative(ok: pd.DataFrame, fit_end: int) -> np.ndarray:
    """Cumulative fleet-wide *relative* km/day by absolute day (month-of-year profile, mean 1).

    Fitted on readings with day <= fit_end only. F[t] - F[l] ~ expected seasonal km factor
    accumulated over days l+1..t.
    """
    s = ok[ok.day <= fit_end]
    g = s.groupby(KEY)
    dkm, dd = g.km.diff(), g.day.diff()
    m = (dd > 0) & (dd <= 60) & (dkm >= 0)
    mid = np.rint((s.day - dd / 2)[m]).astype(int).to_numpy()
    month = MONTH[mid]
    prof = dkm[m].groupby(month).sum() / dd[m].groupby(month).sum()
    prof = prof.reindex(range(1, 13)).fillna(prof.mean())
    rel = (prof / prof.mean()).to_numpy()
    return np.cumsum(rel[MONTH - 1])


def prep_right(ok: pd.DataFrame, vehicles) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History table restricted to `vehicles`, sorted by day (for merge_asof), + first readings."""
    r = ok[ok[KEY].isin(vehicles)].copy()
    r["last_day"] = r.day
    first = r.sort_values("day").groupby(KEY).agg(first_day=("day", "first"), first_km=("km", "first"))
    return r.sort_values("day").reset_index(drop=True), first.reset_index()


def build_samples(right: pd.DataFrame, first: pd.DataFrame, all_: pd.DataFrame, cutoff: int,
                  horizon: int, F: np.ndarray, max_day: int | None = None,
                  max_targets: int | None = None, rng: np.random.Generator | None = None,
                  ok_targets_only: bool = False) -> pd.DataFrame:
    """Samples for one cutoff. `right`/`first` from prep_right; `all_` = all daily readings
    (incl. flagged outliers) of the same vehicles, used only for the targets."""
    vehicles = right[KEY].unique()
    q = pd.DataFrame({KEY: vehicles, "day": np.int32(cutoff)})
    last = pd.merge_asof(q, right, on="day", by=KEY, direction="backward")
    last = last.dropna(subset=["last_day"]).drop(columns="day")
    last["last_day"] = last.last_day.astype("int32")
    last = last.rename(columns={"km": "km0", "cnt": "cnt0"})
    last = last.merge(first, on=KEY, how="left")
    base = right[[KEY, "day", "km", "cnt"]]
    for w in WINDOWS:
        q2 = last[[KEY, "last_day"]].assign(day=(last.last_day - w).astype("int32"))
        q2 = q2.sort_values("day")
        a = pd.merge_asof(q2, base, on="day", by=KEY, direction="backward")
        a = a.set_index(KEY)
        idx = last[KEY]
        span = last.last_day.to_numpy() - a.day.reindex(idx).to_numpy()
        dk = last.km0.to_numpy() - a.km.reindex(idx).to_numpy()
        last[f"rate_{w}"] = np.where(span > 0, dk / np.where(span > 0, span, 1), np.nan)
        if w == 90:
            last["n_90"] = last.cnt0.to_numpy() - a.cnt.reindex(idx).fillna(-1).to_numpy()
    span = (last.last_day - last.first_day).to_numpy()
    last["hist_days"] = span
    last["cum_rate"] = np.where(span > 0, (last.km0 - last.first_km) / np.where(span > 0, span, 1), np.nan)
    last["stale"] = cutoff - last.last_day
    last["rem_to_max"] = last.vhcl_max_mileage - last.km0
    last["ret_in"] = last.return_day - last.last_day
    last["since_handover"] = last.last_day - last.handover_day
    last["last_doy"] = DOY[last.last_day.to_numpy()]
    last = last.drop(columns=["first_day", "first_km", "cnt0", "handover_day", "return_day"])

    t = all_[(all_.day > cutoff) & (all_.day <= (cutoff + horizon if max_day is None
                                                  else min(cutoff + horizon, max_day)))]
    t = t[t[KEY].isin(last[KEY])][[KEY, "day", "km", "ok"]].rename(
        columns={"day": "day_t", "km": "km_t", "ok": "ok_t"})
    if ok_targets_only:
        t = t[t.ok_t]
    if max_targets:
        t = t.assign(_r=rng.random(len(t))).sort_values("_r").groupby(KEY).head(max_targets).drop(columns="_r")
    d = t.merge(last, on=KEY, how="inner")
    dt = d.day_t.to_numpy()
    d["cutoff"] = cutoff
    d["gap"] = dt - d.last_day.to_numpy()
    d["h"] = dt - cutoff
    d["tgt_month"] = MONTH[dt]
    d["tgt_doy"] = DOY[dt]
    d["seas_ratio"] = (F[dt] - F[d.last_day.to_numpy()]) / d.gap
    d["age_t"] = YEAR[dt] - d.model_year
    d["y_rate"] = (d.km_t - d.km0) / d.gap
    d["vehicle_level"] = d.vehicle_level.astype("category")
    return d.reset_index(drop=True)
