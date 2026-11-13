"""Cutoff-based forecast experiment on the real data.

Validation mimics the hidden test task: pick a cutoff C (default 2025-07-01, same season as the
real test cutoff 2026-07-01), hide everything after it for held-out vehicles, and predict every
later daily reading within `val_horizon` days. Training uses only earlier cutoffs, only targets
<= C and only the other vehicles.

Usage: python -m src.run_forecast [--frac 0.5] [--cutoff 2025-07-01]
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import baselines as B
from .features import GroupRates, build_samples, day_of, prep_right, seasonal_cumulative
from .series import HERE, KEY, prepare

NUM = ["km0", "gap", "h", "stale", "rate_7", "rate_30", "rate_90", "rate_180", "rate_365",
       "cum_rate", "n_90", "hist_days", "rem_to_max", "ret_in", "since_handover", "last_doy",
       "tgt_month", "tgt_doy", "seas_ratio", "age_t", "status_code_anonymized", "uti_plan",
       "power_ps", "cubic_capacity_liter", "model_year", "acriss_code_anonymized"]
# group-level / shrunk-rate / rental-behaviour features (PLAN P2)
NEW_NUM = ["g_pool_30", "g_pool_90", "g_region_90", "g_level_90", "g_prod_90", "g_fleet_30",
           "g_fleet_90", "g_fleet_365", "g_pool_ly", "g_fleet_ly", "g_base", "rate_shr",
           "rate_vs_pool", "fleet_trend", "on_share_90", "on_share_365", "lt_share_90",
           "lt_share_365", "km_frac_max", "ret_in_gap"]
# rntl_return_date is the ACTUAL return date => future information for rentals still open at the
# cutoff; excluded from every model (PLAN.md section 14). Kept only for the --with_leaky ablation.
LEAKY = ["ret_in", "ret_in_gap"]
NEW_NUM = [f for f in NEW_NUM if f not in LEAKY]
NUM = [f for f in NUM if f not in LEAKY]
CAT = ["vehicle_level", "vehicle_type", "driven_wheels", "on_rent_status", "rental_product",
       "one_way_flg", "repair_status", "brnc_region_code", "brnc_state_code", "brnc_pool_code"]
FEATURES = NUM + CAT          # default set: base features only (no leaky, no group-rate extras)
BUCKETS = [(1, 14), (15, 30), (31, 60), (61, 90), (91, 150)]
pd.set_option("display.width", 220, "display.max_columns", 30)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metric_row(d: pd.DataFrame, col: str) -> dict:
    err = d[col] - d.km_t
    denom = (d.km_t - d.km0).abs().sum()
    w = 1.0 / d.groupby(KEY)[KEY].transform("size")     # every vehicle counts once
    return {"n": len(d), "MAE": err.abs().mean(), "MedAE": err.abs().median(),
            "WAPE": err.abs().sum() / denom if denom > 0 else np.nan, "bias": err.mean(),
            "MAE_vw": (err.abs() * w).sum() / w.sum()}


def metric_table(val: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        for name, lo, hi in [("all", 1, 10_000)] + [(f"{a}-{b}", a, b) for a, b in BUCKETS]:
            sub = val[(val.gap >= lo) & (val.gap <= hi)]
            rows.append({"model": m, "gap_days": name, **metric_row(sub, m)})
    return pd.DataFrame(rows).set_index(["model", "gap_days"])


def make_train(ok, all_, vehicles, cutoffs, c_val, F, args, rng, grp=None):
    right, first = prep_right(ok, vehicles)
    all_v = all_[all_[KEY].isin(vehicles)]
    parts = [build_samples(right, first, all_v, c, 150, F, max_day=c_val,
                           max_targets=args.max_targets, rng=rng, ok_targets_only=True, grp=grp)
             for c in cutoffs]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2025-07-01")
    ap.add_argument("--val_end", default="2025-11-11")
    ap.add_argument("--frac", type=float, default=0.5, help="share of training vehicles used")
    ap.add_argument("--max_targets", type=int, default=8,
                    help="targets sampled per vehicle and cutoff (0 = all); rows are re-weighted")
    ap.add_argument("--with_new", action="store_true", help="add group-rate/shrunk-rate features")
    ap.add_argument("--with_leaky", action="store_true", help="add ret_in features (leakage, ablation only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default="", help="comma-separated features to drop (ablation)")
    ap.add_argument("--step", type=int, default=30, help="days between training cutoffs")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    ap.add_argument("--params", default="", help="JSON from src.hpo (uses its final_params)")
    args = ap.parse_args()
    feats = FEATURES + (NEW_NUM if args.with_new else []) + (LEAKY if args.with_leaky else [])
    feats = [f for f in feats if f not in args.drop.split(",")]
    rng = np.random.default_rng(args.seed)

    log("loading + cleaning")
    all_, ok = prepare(HERE / "data" / "data_30_percent_sample.parquet")
    log(f"{len(all_):,} daily readings, {all_.ok.mean():.4f} ok, {all_[KEY].nunique():,} vehicles")
    c_val, v_end = day_of(args.cutoff), day_of(args.val_end)

    vids = np.sort(all_[KEY].unique())
    rng.shuffle(vids)
    n_val = int(0.2 * len(vids))
    val_v, rest = vids[:n_val], vids[n_val:]
    n_inner = int(0.1 * len(rest))
    inner_v = rest[:n_inner]
    train_v = rest[n_inner:int(n_inner + (len(rest) - n_inner) * args.frac)]
    log(f"vehicles: train {len(train_v):,}, inner-val {len(inner_v):,}, held-out val {len(val_v):,}")

    F = seasonal_cumulative(ok[ok[KEY].isin(np.concatenate([train_v, inner_v]))], c_val)
    cutoffs = list(range(day_of("2023-03-01"), c_val - 7, args.step))
    log(f"building training samples for {len(cutoffs)} cutoffs")
    log("group-level rate tables")
    grp = GroupRates([ok]) if args.with_new else None
    tr = make_train(ok, all_, train_v, cutoffs, c_val, F, args, rng, grp)
    inner = make_train(ok, all_, inner_v, cutoffs, c_val, F, args, rng, grp)
    right, first = prep_right(ok, val_v)
    val = build_samples(right, first, all_[all_[KEY].isin(val_v)], c_val, v_end - c_val, F, grp=grp)
    log(f"train {len(tr):,} rows, inner {len(inner):,}, val {len(val):,} "
        f"(val targets ok: {val.ok_t.mean():.4f}, vehicles with history: {val[KEY].nunique():,})")

    # ---- baselines (fitted/evaluated without the held-out vehicles)
    val["rate90"] = val.rate_90
    seg_tl = ok[ok[KEY].isin(train_v) & (ok.day <= c_val)].copy()
    segs = ["vehicle_level", "vehicle_type"]
    seg_tl[segs] = seg_tl[segs].astype(str)
    seg, glob = B.fit_segment_rates(seg_tl)
    pv = val.copy()
    pv[segs] = pv[segs].astype(str)
    val["b0_carry"] = B.b0_carry_forward(val)
    val["b1_cum_rate"] = B.b1_cumulative_rate(val)
    val["b1b_rate90"] = B.b1b_recent_rate(val)
    val["b2_segment"] = B.b2_segment_rate(pv, seg, glob).to_numpy()
    r = val.rate_90.fillna(val.cum_rate).fillna(glob)
    val["b3_rate90_season"] = val.km0 + r * val.gap * val.seas_ratio

    # ---- LightGBM on rate, weighted by gap => minimises MAE in km
    log("training LightGBM")
    params = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0)
    if args.params:
        import json
        params.update(json.load(open(args.params))["final_params"])
        log(f"tuned params from {args.params}")
    mdl = lgb.LGBMRegressor(n_estimators=4000, n_jobs=16, verbose=-1, random_state=args.seed, **params)
    mdl.fit(tr[feats], tr.y_rate, sample_weight=tr.gap * tr.w_tgt,
            eval_set=[(inner[feats], inner.y_rate)], eval_sample_weight=[inner.gap * inner.w_tgt],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    log(f"best iteration {mdl.best_iteration_}")
    cap = float(tr.y_rate.quantile(0.999))
    val["m_lgbm"] = val.km0 + np.clip(mdl.predict(val[feats]), 0, cap) * val.gap

    # per-gap-bucket multiplicative bias calibration, fitted on the vehicle-disjoint inner slice
    edges = [b[1] for b in BUCKETS[:-1]]
    d_in = np.clip(mdl.predict(inner[feats]), 0, cap) * inner.gap.to_numpy()
    t_in = inner.y_rate.to_numpy() * inner.gap.to_numpy()
    b_in, b_val = np.digitize(inner.gap, edges, right=True), np.digitize(val.gap, edges, right=True)
    grid = np.linspace(0.9, 1.3, 41)
    w_in = inner.w_tgt.to_numpy()
    k = [grid[np.argmin([np.average(np.abs(g * d_in[b_in == b] - t_in[b_in == b]), weights=w_in[b_in == b])
                         for g in grid])] for b in range(len(BUCKETS))]
    log(f"calibration factors per gap bucket: {np.round(k, 3).tolist()}")
    d_val = (val.m_lgbm - val.km0).to_numpy()
    val["m_lgbm_cal"] = val.km0 + d_val * np.asarray(k)[b_val]

    models = ["b0_carry", "b1_cum_rate", "b1b_rate90", "b2_segment", "b3_rate90_season",
              "m_lgbm", "m_lgbm_cal"]
    for title, sub in [("ALL target readings (as the hidden truth will be: includes noisy ones)", val),
                       ("only target readings passing the outlier filter", val[val.ok_t])]:
        tab = metric_table(sub, models)
        print(f"\n=== {title} ===\n{tab.round(3).to_string()}")
    out = metric_table(val, models).assign(targets="all")
    out = pd.concat([out, metric_table(val[val.ok_t], models).assign(targets="ok_only")])
    (HERE / "results").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / f"forecast_val{args.tag}.csv")
    print("\nSUMMARY", args.tag, "drop=", args.drop, "frac=", args.frac, "step=", args.step)
    print(out.xs("all", level="gap_days").round(3).to_string())
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\nTop features (gain):\n", (imp / imp.sum()).round(3).head(15).to_string())


if __name__ == "__main__":
    main()
