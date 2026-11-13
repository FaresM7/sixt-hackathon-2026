"""Final fit on ALL labelled vehicles + predictions for the masked rows of test_data.parquet.

Validation (src.run_forecast) holds out vehicles and a time window. Here nothing is held out for
scoring: the cutoff is the real one (2026-07-01) and the targets are the blank test rows, whose
true values we do not have. A 10% vehicle slice is used only to pick the number of trees, then
the model is refitted on everything with that number of trees.

Usage: python -m src.train_final [--step 30] [--max_targets 8] [--seed 0]
       [--params results/hpo/<tag>_best.json]   (tuned LightGBM params from src.hpo)
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .coldstart import COLD_FEATS, cold_features, cold_samples, fit_cold
from .features import build_samples, day_of, prep_right, seasonal_cumulative
from .run_forecast import FEATURES, log, make_train
from .series import (
    EPOCH,
    HERE,
    KEY,
    STATIC_COLS,
    daily_series,
    flag_outliers,
    load_parquet,
    ok_series,
    prepare,
)

TRAIN = HERE / "data" / "data_30_percent_sample.parquet"
TEST = HERE / "data" / "test_data.parquet"
PARAMS = {"objective": "l1", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 200,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 5.0,
              "n_jobs": 16, "verbose": -1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--max_targets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final_cutoff", default="2026-07-01")
    ap.add_argument("--params", default="", help="JSON from src.hpo (uses its final_params)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    c_fin = day_of(args.final_cutoff)
    params = dict(PARAMS)
    if args.params:
        params.update(json.loads(Path(args.params).read_text(encoding="utf-8"))["final_params"])
        log(f"using tuned params from {args.params}: {params}")

    log("loading labelled train file")
    all_, ok = prepare(TRAIN)
    vids = np.sort(all_[KEY].unique())
    rng.shuffle(vids)
    inner_v, fit_v = vids[: int(0.1 * len(vids))], vids[int(0.1 * len(vids)):]
    F = seasonal_cumulative(ok, c_fin)
    cutoffs = list(range(day_of("2023-03-01"), c_fin - 7, args.step))
    log(f"{len(vids):,} vehicles, {len(cutoffs)} cutoffs, final cutoff {args.final_cutoff}")

    # ---- main model: pick #trees on a vehicle-disjoint slice, refit on everything
    tr = make_train(ok, all_, fit_v, cutoffs, c_fin, F, args, rng)
    inner = make_train(ok, all_, inner_v, cutoffs, c_fin, F, args, rng)
    log(f"samples: fit {len(tr):,}, inner {len(inner):,}")
    m = lgb.LGBMRegressor(n_estimators=4000, random_state=args.seed, **params)
    m.fit(tr[FEATURES], tr.y_rate, sample_weight=tr.gap * tr.w_tgt, eval_set=[(inner[FEATURES], inner.y_rate)],
          eval_sample_weight=[inner.gap * inner.w_tgt], callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees = int(m.best_iteration_ * 1.1)
    log(f"best iteration {m.best_iteration_} -> refit on all vehicles with {n_trees} trees")
    full = pd.concat([tr, inner], ignore_index=True)
    model = lgb.LGBMRegressor(n_estimators=n_trees, random_state=args.seed, **params)
    model.fit(full[FEATURES], full.y_rate, sample_weight=full.gap * full.w_tgt)
    rate_cap = float(full.y_rate.quantile(0.999))

    # ---- cold-start model (vehicles without any reading before the cutoff)
    cold_tr = cold_samples(all_, vids, cutoffs, max_day=c_fin, rng=rng, max_targets=8)
    cold = fit_cold(cold_tr, args.seed)
    log(f"cold-start model fitted on {len(cold_tr):,} rows from {cold_tr[KEY].nunique():,} vehicles")

    # ---- test file: known history + masked target rows
    log("loading test file")
    raw = load_parquet(TEST)
    ids = pd.read_parquet(TEST, columns=[KEY])[KEY]
    uniques = pd.factorize(ids)[1]               # same order as load_parquet's integer codes
    masked = raw[raw.vhcl_current_mileage.isna()]
    tgt = masked[[KEY, "day"] + STATIC_COLS].copy()
    s_t = flag_outliers(daily_series(raw))
    ok_t = ok_series(s_t)
    has_hist = np.intersect1d(tgt[KEY].unique(), ok_t[KEY].unique())
    log(f"masked rows {len(tgt):,}; vehicles {tgt[KEY].nunique():,}; with history {len(has_hist):,}")

    right, first = prep_right(ok_t, has_hist)
    tq = tgt[tgt[KEY].isin(has_hist)].assign(km=np.nan, ok=True)[[KEY, "day", "km", "ok"]]
    X = build_samples(right, first, tq, c_fin, 400, F)
    rate = np.clip(model.predict(X[FEATURES]), 0, rate_cap)
    out = pd.DataFrame({KEY: X[KEY], "day": X.day_t, "km_pred": X.km0 + rate * X.gap,
                        "gap": X.gap, "source": "history"})

    cold_rows = tgt[~tgt[KEY].isin(has_hist)]
    if len(cold_rows):
        Xc = cold_features(cold_rows, c_fin)
        out_c = pd.DataFrame({KEY: Xc[KEY], "day": Xc.day, "km_pred": np.maximum(cold.predict(Xc[COLD_FEATS]), 0),
                              "gap": np.nan, "source": "cold_start"})
        out = pd.concat([out, out_c], ignore_index=True)

    out = out.sort_values([KEY, "day"])
    out["km_pred"] = out.groupby(KEY).km_pred.cummax().round(0)   # odometer never decreases
    h = out[out.source == "history"].merge(X[[KEY, "km0", "last_day"]].drop_duplicates(KEY), on=KEY)
    imp_rate = (h.km_pred - h.km0) / (h.day - h.last_day)
    out[KEY] = uniques[out[KEY].to_numpy()]       # back to the original anonymised ids
    out["date"] = (EPOCH + pd.to_timedelta(out.day, unit="D")).dt.date
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    out[["vhcl_anon_id", "date", "km_pred", "source", "gap"]].to_csv(res / "test_predictions.csv", index=False)

    log(f"wrote {len(out):,} predictions ({(out.source == 'cold_start').mean():.3%} cold start)")
    print("implied km/day of predictions (history rows):", imp_rate.describe().round(1).to_dict())
    print("km_pred quantiles:", out.km_pred.quantile([0, .01, .5, .99, 1]).round(0).to_dict())
    print("missing predictions:", len(tgt) - len(out))


if __name__ == "__main__":
    main()
