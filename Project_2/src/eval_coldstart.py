"""Validation of the cold-start model (vehicles with NO reading before the cutoff).

Same held-out vehicles and cutoff as run_forecast: among the held-out vehicles, those whose first
reading lies after 2025-07-01; targets = their readings until 2025-11-11. The model only knows
static attributes + the target date (as for the 149 test vehicles without history).
Baselines: median km of new cars at the same horizon (h bucket), and the same per vehicle_level.

Usage: python -m src.eval_coldstart
"""
import argparse

import numpy as np
import pandas as pd

from .coldstart import COLD_FEATS, cold_samples, fit_cold
from .features import day_of
from .run_forecast import log
from .series import HERE, KEY, prepare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2025-07-01")
    ap.add_argument("--val_end", default="2025-11-11")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    all_, _ok = prepare(HERE / "data" / "data_30_percent_sample.parquet")
    vids = np.sort(all_[KEY].unique())
    rng.shuffle(vids)                                   # same split as run_forecast (seed 0)
    n_val = int(0.2 * len(vids))
    val_v, rest = vids[:n_val], vids[n_val:]
    c_val, v_end = day_of(args.cutoff), day_of(args.val_end)
    cutoffs = list(range(day_of("2023-03-01"), c_val - 7, 30))

    tr = cold_samples(all_, rest, cutoffs, max_day=c_val, rng=rng, max_targets=8)
    val = cold_samples(all_, val_v, [c_val], horizon=v_end - c_val, max_targets=0)
    log(f"train {len(tr):,} rows ({tr[KEY].nunique():,} vehicles); held-out new cars: "
        f"{len(val):,} readings of {val[KEY].nunique():,} vehicles")
    m = fit_cold(tr, args.seed)
    val["cold_model"] = np.maximum(m.predict(val[COLD_FEATS]), 0)

    bins = [0, 15, 30, 60, 90, 150, 10_000]
    for d in (tr, val):
        d["hb"] = pd.cut(d.h, bins).astype(str)
    glob = tr.groupby("hb").km.median()
    lvl = tr.groupby(["vehicle_level", "hb"], observed=True).km.median()
    val["base_h"] = val.hb.map(glob)
    idx = pd.MultiIndex.from_arrays([val.vehicle_level.astype(str), val.hb])
    lvl.index = pd.MultiIndex.from_arrays([lvl.index.get_level_values(0).astype(str), lvl.index.get_level_values(1)])
    val["base_level_h"] = pd.Series(lvl.reindex(idx).to_numpy(), index=val.index).fillna(val.base_h)

    rows = []
    for name in ["base_h", "base_level_h", "cold_model"]:
        e = val[name] - val.km
        rows.append({"model": name, "n": len(val), "MAE": e.abs().mean(), "MedAE": e.abs().median(),
                     "bias": e.mean(), "median_true_km": val.km.median()})
    out = pd.DataFrame(rows).set_index("model")
    print(out.round(1).to_string())
    out.to_csv(HERE / "results" / "coldstart_val.csv")


if __name__ == "__main__":
    main()
