"""Model-family comparison on the same split/samples as run_forecast (for the SUBMISSION table).

Models (all predict the km/day rate with sample weight gap*w_tgt, so the loss is row-weighted
km error): Ridge (linear reference), sklearn HistGradientBoosting (L1), LightGBM (L1, our choice).
Ensembles: seed-bagged LightGBM, plain means, and a convex blend whose weights are fitted on the
vehicle-disjoint inner slice (full-horizon cutoffs 150-365 d before the val cutoff), never on val.
Usage: python -m src.compare_models [--frac 0.5] [--bags 3]
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import build_samples, day_of, prep_right, seasonal_cumulative
from .run_forecast import CAT, FEATURES, NUM, log, make_train, metric_row
from .series import HERE, KEY, prepare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--max_targets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bags", type=int, default=3, help="extra LightGBM seeds for bagging")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    all_, ok = prepare(HERE / "data" / "data_30_percent_sample.parquet")
    c_val, v_end = day_of("2025-07-01"), day_of("2025-11-11")
    vids = np.sort(all_[KEY].unique())
    rng.shuffle(vids)                                    # same split as run_forecast (seed 0)
    n_val = int(0.2 * len(vids))
    val_v, rest = vids[:n_val], vids[n_val:]
    n_inner = int(0.1 * len(rest))
    inner_v = rest[:n_inner]
    train_v = rest[n_inner:int(n_inner + (len(rest) - n_inner) * args.frac)]
    F = seasonal_cumulative(ok[ok[KEY].isin(np.concatenate([train_v, inner_v]))], c_val)
    cutoffs = list(range(day_of("2023-03-01"), c_val - 7, 30))
    tr = make_train(ok, all_, train_v, cutoffs, c_val, F, args, rng)
    inner = make_train(ok, all_, inner_v, cutoffs, c_val, F, args, rng)
    right, first = prep_right(ok, val_v)
    val = build_samples(right, first, all_[all_[KEY].isin(val_v)], c_val, v_end - c_val, F)
    log(f"train {len(tr):,}, val {len(val):,}")
    w = (tr.gap * tr.w_tgt).to_numpy()
    cap = float(tr.y_rate.quantile(0.999))
    rows = {}
    blend = inner[(inner.cutoff >= c_val - 365) & (inner.cutoff <= c_val - 150)]
    inner_rate = {}                                       # rate predictions on the blend slice

    def record(name, rate_hat, secs):
        val[name] = val.km0 + np.clip(rate_hat, 0, cap) * val.gap
        rows[name] = {**metric_row(val, name), "fit_seconds": round(secs)}
        log(f"{name}: MAE {rows[name]['MAE']:.0f}  ({secs:.0f}s)")

    # --- Ridge on scaled numerics + one-hot (rare categories pooled), median-imputed
    t0 = time.time()
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200), CAT)])
    ridge = make_pipeline(pre, Ridge(alpha=10.0))
    s_ = lambda d, c: d[c].astype(object).where(d[c].notna(), "NA").astype(str)
    Xtr = tr[FEATURES].assign(**{c: s_(tr, c) for c in CAT})
    Xv = val[FEATURES].assign(**{c: s_(val, c) for c in CAT})
    ridge.fit(Xtr, tr.y_rate, ridge__sample_weight=w)
    record("ridge", ridge.predict(Xv), time.time() - t0)
    inner_rate["ridge"] = ridge.predict(blend[FEATURES].assign(**{c: s_(blend, c) for c in CAT}))

    # --- sklearn HistGradientBoosting (L1) with native categoricals
    t0 = time.time()
    st = lambda d, c: d[c].astype(object).where(d[c].notna(), "NA").astype(str)
    cats = {c: pd.CategoricalDtype(sorted(st(tr, c).unique())) for c in CAT}
    enc = lambda d: d[FEATURES].assign(**{c: st(d, c).astype(cats[c]).cat.codes for c in CAT})
    hgb = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.1, max_iter=400,
                                        max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=5.0,
                                        categorical_features=[FEATURES.index(c) for c in CAT],
                                        early_stopping=True, random_state=args.seed)
    hgb.fit(enc(tr), tr.y_rate, sample_weight=w)
    record("hist_gbm", hgb.predict(enc(val)), time.time() - t0)
    inner_rate["hist_gbm"] = hgb.predict(enc(blend))

    # --- LightGBM (our model)
    t0 = time.time()
    m = lgb.LGBMRegressor(objective="l1", n_estimators=3000, learning_rate=0.05, num_leaves=63,
                          min_child_samples=200, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=5.0, n_jobs=16, verbose=-1,
                          random_state=args.seed)
    m.fit(tr[FEATURES], tr.y_rate, sample_weight=w, eval_set=[(inner[FEATURES], inner.y_rate)],
          eval_sample_weight=[(inner.gap * inner.w_tgt).to_numpy()],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    record("lightgbm", m.predict(val[FEATURES]), time.time() - t0)
    inner_rate["lightgbm"] = m.predict(blend[FEATURES])

    # --- seed bagging: same params, different seeds (row/feature subsampling differs)
    t0 = time.time()
    v_bag, i_bag = [m.predict(val[FEATURES])], [inner_rate["lightgbm"]]
    for k in range(1, args.bags + 1):
        mk = lgb.LGBMRegressor(**{**m.get_params(), "random_state": args.seed + k,
                                  "n_estimators": m.best_iteration_})
        mk.fit(tr[FEATURES], tr.y_rate, sample_weight=w)
        v_bag.append(mk.predict(val[FEATURES]))
        i_bag.append(mk.predict(blend[FEATURES]))
    record(f"lgbm_bag{args.bags + 1}", np.mean(v_bag, axis=0), time.time() - t0)
    inner_rate[f"lgbm_bag{args.bags + 1}"] = np.mean(i_bag, axis=0)

    # --- ensembles (val predictions come from the per-model columns, in km)
    bag = f"lgbm_bag{args.bags + 1}"
    vr = {n: ((val[n] - val.km0) / val.gap).to_numpy() for n in ["ridge", "hist_gbm", bag]}
    record("mean_gbms", (vr["hist_gbm"] + vr[bag]) / 2, 0)
    record("mean_all3", (vr["ridge"] + vr["hist_gbm"] + vr[bag]) / 3, 0)
    ir = {n: np.clip(inner_rate[n], 0, cap) for n in ["ridge", "hist_gbm", bag]}
    g, wt = blend.gap.to_numpy(), blend.w_tgt.to_numpy()
    yt = blend.y_rate.to_numpy()
    best = (np.inf, None)
    for a in np.arange(0, 1.0001, 0.05):            # convex weights (ridge, hist_gbm, bag)
        for b in np.arange(0, 1.0001 - a, 0.05):
            r = a * ir["ridge"] + b * ir["hist_gbm"] + (1 - a - b) * ir[bag]
            e = np.average(np.abs(r - yt) * g, weights=wt)
            if e < best[0]:
                best = (e, (a, b, 1 - a - b))
    a, b, c = best[1]
    log(f"blend weights fitted on inner: ridge {a:.2f}, hist_gbm {b:.2f}, {bag} {c:.2f}")
    record("blend_inner_w", a * vr["ridge"] + b * vr["hist_gbm"] + c * vr[bag], 0)

    out = pd.DataFrame(rows).T[["n", "MAE", "MAE_vw", "MedAE", "WAPE", "bias", "fit_seconds"]]
    print(out.round(3).to_string())
    out.to_csv(HERE / "results" / "model_comparison.csv")


if __name__ == "__main__":
    main()
