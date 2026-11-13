"""Loading and cleaning: raw status rows -> a per-vehicle odometer timeline.

Row semantics (verified on example_file.csv, to be confirmed on the full data):
  * OnRent rows carry the odometer at rental HANDOVER (constant over the whole
    rental, even for mid-rental snapshots). One 'start' observation per rental.
  * NotOnRent rows carry a fresh reading taken on `vhcl_fct_taken_datm`
    ('idle' observation).
"""
from pathlib import Path

import numpy as np
import pandas as pd

KEY = "vhcl_anon_id"
DATE_COLS = ["vhcl_fct_taken_datm", "rntl_handover_date", "rntl_return_date"]
STATIC_COLS = ["acriss_code_anonymized", "vehicle_level", "vehicle_type", "model_year",
               "driven_wheels", "power_ps", "cubic_capacity_liter"]
HERE = Path(__file__).resolve().parent.parent


def default_path() -> Path:
    """First CSV in data/, else the small example file."""
    found = sorted((HERE / "data").glob("*.csv"))
    return found[0] if found else HERE / "example_file.csv"


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV, parse dates, drop rows without mileage/timestamp and exact duplicates."""
    df = pd.read_csv(path or default_path(), parse_dates=DATE_COLS)
    df = df.dropna(subset=["vhcl_current_mileage", "vhcl_fct_taken_datm"])
    return df.drop_duplicates().reset_index(drop=True)


def build_timeline(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (vehicle, date, km) odometer observation, sorted, with a `kind` column."""
    on = raw[(raw.on_rent_status == "OnRent") & raw.rntl_handover_date.notna()]
    starts = (on.groupby([KEY, "rntl_handover_date", "rntl_return_date"], as_index=False)
                .agg(km=("vhcl_current_mileage", "median"),
                     n_km_values=("vhcl_current_mileage", "nunique"),
                     rental_product=("rental_product", "first"),
                     one_way_flg=("one_way_flg", "first")))
    starts = starts.rename(columns={"rntl_handover_date": "date", "rntl_return_date": "rental_return"})
    starts["kind"] = "start"
    idle = raw[raw.on_rent_status != "OnRent"]
    idle = idle.rename(columns={"vhcl_fct_taken_datm": "date", "vhcl_current_mileage": "km"})
    idle = idle[[KEY, "date", "km", "rental_product", "one_way_flg"]].assign(kind="idle")
    tl = pd.concat([starts, idle], ignore_index=True)
    tl["date"] = tl["date"].dt.normalize()
    tl = tl.drop_duplicates([KEY, "date", "km"])
    static = raw.sort_values("vhcl_fct_taken_datm").groupby(KEY)[STATIC_COLS].first().reset_index()
    tl = tl.merge(static, on=KEY, how="left")
    tl = tl.sort_values([KEY, "date", "km"]).reset_index(drop=True)
    tl["day"] = (tl["date"] - pd.Timestamp("2000-01-01")).dt.days
    return tl


def quality_report(tl: pd.DataFrame, max_rate: float = 1000.0) -> dict:
    """Counts of physically suspicious things in the timeline (the future rule layer)."""
    g = tl.groupby(KEY)
    dkm, dday = g.km.diff(), g.day.diff()
    rate = dkm / dday.replace(0, np.nan)
    multi = tl["n_km_values"] if "n_km_values" in tl else pd.Series(dtype=float)
    return {
        "vehicles": tl[KEY].nunique(),
        "observations": len(tl),
        "obs_per_vehicle_median": float(g.size().median()),
        "mileage_decreases": int((dkm < 0).sum()),
        "same_day_conflicts": int(((dday == 0) & (dkm != 0)).sum()),
        "rate_above_cap": int((rate > max_rate).sum()),
        "rental_with_multiple_km_values": int((multi > 1).sum()),
        "date_min": str(tl.date.min().date()),
        "date_max": str(tl.date.max().date()),
    }
