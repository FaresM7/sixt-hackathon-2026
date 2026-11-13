"""Compute baseline results. Usage: python -m src.run_baselines [file.csv]"""
import sys
from pathlib import Path

import pandas as pd

from . import baselines as B
from .data import HERE, KEY, build_timeline, load_raw, quality_report
from .evaluate import make_pairs, metrics_table, time_cutoff, vehicle_split

pd.set_option("display.width", 200)


def score(pairs: pd.DataFrame, train_tl: pd.DataFrame) -> pd.DataFrame:
    """Score all baselines on `pairs`; the segment baseline is fitted on `train_tl` only."""
    seg, glob = B.fit_segment_rates(train_tl)
    pairs = pairs.copy()
    pairs["b0_carry"] = B.b0_carry_forward(pairs)
    pairs["b1_cum_rate"] = B.b1_cumulative_rate(pairs)
    pairs["b1b_rate90"] = B.b1b_recent_rate(pairs)
    pairs["b2_segment"] = B.b2_segment_rate(pairs, seg, glob)
    return metrics_table(pairs, ["b0_carry", "b1_cum_rate", "b1b_rate90", "b2_segment"])


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    tl = build_timeline(load_raw(path))
    print("DATA QUALITY:", quality_report(tl))
    pairs = make_pairs(tl)
    cut = time_cutoff(tl)
    vids = tl[KEY].unique()
    results = {}
    if len(vids) >= 5:
        train_v, test_v = vehicle_split(vids)
        in_tr, in_te = tl[KEY].isin(train_v), pairs.vid.isin(test_v)
        results["held-out vehicles (all time)"] = score(pairs[in_te], tl[in_tr])
        results["held-out vehicles + after cutoff"] = score(
            pairs[in_te & (pairs.day_t >= cut)], tl[in_tr & (tl.day < cut)])
    else:
        print(f"NOTE: only {len(vids)} vehicle(s): no vehicle split; b2 is fitted in-sample.")
    results["time split (all vehicles, targets after cutoff)"] = score(
        pairs[pairs.day_t >= cut], tl[tl.day < cut])
    results["no split (all pairs; b2 in-sample)"] = score(pairs, tl)
    out = []
    for name, tab in results.items():
        print(f"\n=== {name} ===\n{tab.round(3).to_string()}")
        out.append(tab.assign(regime=name))
    (HERE / "results").mkdir(exist_ok=True)
    pd.concat(out).to_csv(HERE / "results" / "baseline_metrics.csv")


if __name__ == "__main__":
    main()
