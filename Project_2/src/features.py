"""Cutoff-based forecasting samples.

A sample = (vehicle, cutoff C, target day t > C). Everything known at C is the vehicle's readings
up to C (its last known reading may be older than C: reporting lag). Target = km at day t.
All features use readings with day <= C only (no leakage); the seasonal table is fitted on an
explicitly given period.
"""
import numpy as np
import pandas as pd

from .series import EPOCH, KEY

WINDOWS = (7, 30, 90, 180, 365)
SHRINK_K = 4.0   # pseudo-months of group-level evidence in the shrunk vehicle rate
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


class GroupRates:
    """Time-safe group-level usage rates (pooled km/day = sum dkm / sum dday over consecutive
    clean readings). `at(cutoff)` only uses intervals ending on/before the cutoff.

    Groups: branch pool / region, vehicle level, rental product of the interval's end reading,
    whole fleet. Keys are strings so tables can combine several parquet files.
    """
    MIN_DAYS = 100   # minimum observed vehicle-days for a group rate to be trusted

    def __init__(self, oks: list[pd.DataFrame]):
        parts = []
        for ok in oks:
            g = ok.groupby(KEY)
            iv = pd.DataFrame({"day": ok.day, "dkm": g.km.diff(), "dday": g.day.diff(),
                               "pool": ok.brnc_pool_code.astype(str),
                               "region": ok.brnc_region_code.astype(str),
                               "level": ok.vehicle_level.astype(str),
                               "prod": ok.rental_product.astype(str)})
            parts.append(iv[(iv.dday > 0) & (iv.dday <= 60) & (iv.dkm >= 0)])
        iv = pd.concat(parts, ignore_index=True).sort_values("day").reset_index(drop=True)
        iv["fleet"] = "all"
        iv["ym"] = (YEAR[iv.day.to_numpy()] * 12 + MONTH[iv.day.to_numpy()] - 1)
        self.iv, self.day = iv, iv.day.to_numpy()
        m = iv.groupby(["pool", "ym"])[["dkm", "dday"]].sum()
        self.monthly_pool = (m.dkm / m.dday).where(m.dday >= self.MIN_DAYS).rename("g_pool_ly")
        f = iv.groupby("ym")[["dkm", "dday"]].sum()
        self.monthly_fleet = (f.dkm / f.dday).rename("g_fleet_ly")

    def at(self, cutoff: int) -> dict[str, pd.Series]:
        """{feature name: Series indexed by group key} using intervals in (cutoff - W, cutoff]."""
        out = {}
        for col, wins in [("pool", (30, 90)), ("region", (90,)), ("level", (90,)),
                          ("prod", (90,)), ("fleet", (30, 90, 365))]:
            for w in wins:
                lo = np.searchsorted(self.day, cutoff - w, side="right")
                hi = np.searchsorted(self.day, cutoff, side="right")
                s = self.iv.iloc[lo:hi].groupby(col)[["dkm", "dday"]].sum()
                out[f"g_{col}_{w}"] = (s.dkm / s.dday).where(s.dday >= self.MIN_DAYS)
        return out


def _by_group(cat: pd.Series, table: pd.Series) -> np.ndarray:
    """Look up a per-group value for every row of a categorical/str column (NaN if unknown)."""
    return cat.astype(str).map(table).to_numpy(dtype=float)


def prep_right(ok: pd.DataFrame, vehicles) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History table restricted to `vehicles`, sorted by day (for merge_asof), + first readings."""
    r = ok[ok[KEY].isin(vehicles)].copy()
    r["cum_on"] = (r.on_rent_status == "OnRent").astype("int32").groupby(r[KEY]).cumsum()
    r["cum_lt"] = (r.rental_product == "LongTerm").astype("int32").groupby(r[KEY]).cumsum()
    r["last_day"] = r.day
    first = r.sort_values("day").groupby(KEY).agg(first_day=("day", "first"), first_km=("km", "first"))
    return r.sort_values("day").reset_index(drop=True), first.reset_index()


def build_samples(right: pd.DataFrame, first: pd.DataFrame, all_: pd.DataFrame, cutoff: int,
                  horizon: int, F: np.ndarray, max_day: int | None = None,
                  max_targets: int | None = None, rng: np.random.Generator | None = None,
                  ok_targets_only: bool = False, grp: "GroupRates | None" = None) -> pd.DataFrame:
    """Samples for one cutoff. `right`/`first` from prep_right; `all_` = all daily readings
    (incl. flagged outliers) of the same vehicles, used only for the targets. `grp` adds
    group-level rate features (pool/region/level/product/fleet) computed up to the cutoff."""
    vehicles = right[KEY].unique()
    q = pd.DataFrame({KEY: vehicles, "day": np.int32(cutoff)})
    last = pd.merge_asof(q, right, on="day", by=KEY, direction="backward")
    last = last.dropna(subset=["last_day"]).drop(columns="day")
    last["last_day"] = last.last_day.astype("int32")
    last = last.rename(columns={"km": "km0", "cnt": "cnt0", "cum_on": "on0", "cum_lt": "lt0"})
    last = last.merge(first, on=KEY, how="left")
    base = right[[KEY, "day", "km", "cnt", "cum_on", "cum_lt"]]
    for w in WINDOWS:
        q2 = last[[KEY, "last_day"]].assign(day=(last.last_day - w).astype("int32"))
        q2 = q2.sort_values("day")
        a = pd.merge_asof(q2, base, on="day", by=KEY, direction="backward")
        a = a.set_index(KEY)
        idx = last[KEY]
        span = last.last_day.to_numpy() - a.day.reindex(idx).to_numpy()
        dk = last.km0.to_numpy() - a.km.reindex(idx).to_numpy()
        last[f"rate_{w}"] = np.where(span > 0, dk / np.where(span > 0, span, 1), np.nan)
        if w in (90, 365):
            n = last.cnt0.to_numpy() - a.cnt.reindex(idx).fillna(-1).to_numpy()
            if w == 90:
                last["n_90"] = n
            on = last.on0.to_numpy() - a.cum_on.reindex(idx).fillna(0).to_numpy()
            lt = last.lt0.to_numpy() - a.cum_lt.reindex(idx).fillna(0).to_numpy()
            last[f"on_share_{w}"] = on / np.maximum(n, 1)
            last[f"lt_share_{w}"] = lt / np.maximum(n, 1)
    span = (last.last_day - last.first_day).to_numpy()
    last["hist_days"] = span
    last["cum_rate"] = np.where(span > 0, (last.km0 - last.first_km) / np.where(span > 0, span, 1), np.nan)
    last["stale"] = cutoff - last.last_day
    last["rem_to_max"] = last.vhcl_max_mileage - last.km0
    last["ret_in"] = last.return_day - last.last_day
    last["since_handover"] = last.last_day - last.handover_day
    last["last_doy"] = DOY[last.last_day.to_numpy()]
    last["km_frac_max"] = last.km0 / last.vhcl_max_mileage
    last = last.drop(columns=["first_day", "first_km", "cnt0", "on0", "lt0", "handover_day", "return_day"])

    t = all_[(all_.day > cutoff) & (all_.day <= (cutoff + horizon if max_day is None
                                                  else min(cutoff + horizon, max_day)))]
    t = t[t[KEY].isin(last[KEY])][[KEY, "day", "km", "ok"]].rename(
        columns={"day": "day_t", "km": "km_t", "ok": "ok_t"})
    if ok_targets_only:
        t = t[t.ok_t]
    if max_targets:
        # sample at most `max_targets` rows per vehicle, but weight each kept row by n_all/n_kept so
        # the training distribution matches row-weighted scoring (busy cars report more often)
        t = t.assign(_r=rng.random(len(t)), _n=t.groupby(KEY)[KEY].transform("size"))
        t = t.sort_values("_r").groupby(KEY).head(max_targets)
        t["w_tgt"] = t._n / np.minimum(t._n, max_targets)
        t = t.drop(columns=["_r", "_n"])
    else:
        t["w_tgt"] = 1.0
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
    d["ret_in_gap"] = np.minimum(d.ret_in, d.gap)
    if grp is not None:
        for name, tab in grp.at(cutoff).items():
            col = {"pool": "brnc_pool_code", "region": "brnc_region_code",
                   "level": "vehicle_level", "prod": "rental_product"}.get(name.split("_")[1])
            d[name] = _by_group(d[col], tab) if col else float(tab.get("all", np.nan))
        ym = (YEAR[dt] - 1) * 12 + MONTH[dt] - 1
        d["g_fleet_ly"] = grp.monthly_fleet.reindex(ym).to_numpy()
        keys = pd.MultiIndex.from_arrays([d.brnc_pool_code.astype(str).to_numpy(), ym])
        d["g_pool_ly"] = grp.monthly_pool.reindex(keys).to_numpy()
        base_rate = d.g_pool_90.fillna(d.g_region_90).fillna(d.g_fleet_90)
        d["g_base"] = base_rate
        vr = d.rate_180.fillna(d.rate_90).fillna(d.cum_rate)
        n_eff = np.minimum(d.hist_days, 180) / 30.0
        d["rate_shr"] = ((n_eff * vr + SHRINK_K * base_rate) / (n_eff + SHRINK_K)).where(vr.notna(), base_rate)
        d["rate_vs_pool"] = d.rate_90 / d.g_pool_90
        d["fleet_trend"] = d.g_fleet_30 / d.g_fleet_365
    d["vehicle_level"] = d.vehicle_level.astype("category")
    return d.reset_index(drop=True)
