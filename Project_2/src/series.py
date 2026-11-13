"""Real-data loading (parquet) -> clean daily odometer series per vehicle.

Pipeline: load_parquet -> daily_series (one reading per vehicle-day: the max) ->
flag_outliers (adds `ok`) -> ok_series (monotone, outlier-free history used for features).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # Project_2/
KEY = "vhcl_anon_id"
EPOCH = pd.Timestamp("2000-01-01")
STATIC_COLS = ["acriss_code_anonymized", "vehicle_level", "vehicle_type", "model_year",
               "driven_wheels", "power_ps", "cubic_capacity_liter"]
STATE_COLS = ["status_code_anonymized", "on_rent_status", "rental_product", "one_way_flg",
              "repair_status", "uti_plan", "vhcl_max_mileage", "brnc_code", "brnc_pool_code",
              "brnc_region_code", "brnc_state_code", "handover_day", "return_day"]
CAT_COLS = ["vehicle_level", "vehicle_type", "driven_wheels", "on_rent_status", "rental_product",
            "one_way_flg", "repair_status", "brnc_code", "brnc_pool_code", "brnc_region_code",
            "brnc_state_code"]
OUTLIER_TOL = 100.0   # km: tolerated non-monotonicity vs the median of neighbouring readings


def to_day(s: pd.Series) -> pd.Series:
    """Datetime-like strings -> integer days since 2000-01-01 (NaN kept as float)."""
    return (pd.to_datetime(s) - EPOCH).dt.days


def load_parquet(path) -> pd.DataFrame:
    """Read a parquet file; vehicle ids -> int, dates -> day numbers, strings -> categories."""
    df = pd.read_parquet(path).drop(columns=["sixt_predicted_mileage_interp", "sixt_error"],
                                    errors="ignore")
    df[KEY] = pd.factorize(df[KEY])[0].astype("int32")
    df["day"] = to_day(df.vhcl_fct_taken_datm).astype("int32")
    df["handover_day"] = to_day(df.rntl_handover_date)
    df["return_day"] = to_day(df.rntl_return_date)
    df = df.drop(columns=["vhcl_fct_taken_datm", "rntl_handover_date", "rntl_return_date"])
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (vehicle, day) with a known mileage: the row with the max km that day."""
    df = df.dropna(subset=["vhcl_current_mileage"]).rename(columns={"vhcl_current_mileage": "km"})
    df = df.sort_values([KEY, "day", "km"]).drop_duplicates([KEY, "day"], keep="last")
    return df[[KEY, "day", "km"] + STATE_COLS + STATIC_COLS].reset_index(drop=True)


def flag_outliers(s: pd.DataFrame) -> pd.DataFrame:
    """Add boolean `ok`: False if a reading breaks monotonicity vs. the median of its 3 previous
    or 3 next readings (by more than OUTLIER_TOL km), or is absurd (>= 900k or > 3x max mileage)."""
    g = s.groupby(KEY).km
    prev = np.column_stack([g.shift(k) for k in (1, 2, 3)])
    nxt = np.column_stack([g.shift(-k) for k in (1, 2, 3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_prev, med_next = np.nanmedian(prev, axis=1), np.nanmedian(nxt, axis=1)
    km = s.km.to_numpy()
    bad = (km < med_prev - OUTLIER_TOL) | (km > med_next + OUTLIER_TOL)
    bad |= (km >= 900_000) | (km > 3 * s.vhcl_max_mileage.to_numpy())
    s = s.copy()
    s["ok"] = ~bad
    return s


def ok_series(s: pd.DataFrame) -> pd.DataFrame:
    """Outlier-free, forced-monotone history with a per-vehicle reading counter `cnt`."""
    o = s[s.ok].copy()
    o["km"] = o.groupby(KEY).km.cummax()
    o["cnt"] = o.groupby(KEY).cumcount().astype("int32")
    return o.reset_index(drop=True)


def prepare(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """parquet -> (all daily readings incl. outliers flagged, clean monotone series)."""
    s = flag_outliers(daily_series(load_parquet(path)))
    return s, ok_series(s)
