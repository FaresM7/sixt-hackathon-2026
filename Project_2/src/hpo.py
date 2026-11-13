"""Hyper-parameter search for the LightGBM rate model (Optuna/TPE, time-capped, resumable).

Search objective: row-weighted MAE in km (weighted L1 on the rate, weight = gap * w_tgt) on a
vehicle-disjoint inner slice, using a subsample of training rows and learning rate 0.1 for speed.
The held-out validation vehicles are only touched at the end: best params vs. default params,
full training rows, lr 0.05, same clipping as run_forecast => numbers comparable to run_forecast.

Inner slice for the search = inner vehicles at cutoffs 150-365 days before the validation cutoff,
i.e. cutoffs whose FULL 150-day horizon lies before the validation cutoff (later cutoffs only have
short-gap targets because targets are capped at the cutoff, which would tune for the wrong gaps).

Usage: python -m src.hpo [--minutes 30] [--rows 600000] [--n_jobs 16] [--tag hpo]
Smoke test: python -m src.hpo --minutes 1 --rows 50000 --frac 0.1 --tag smoke --skip_heldout
Resume: rerun with the same --tag (study stored in results/hpo/<tag>.db).
Use the result: python -m src.train_final --params results/hpo/<tag>_best.json
"""
import argparse
import json
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from .features import build_samples, day_of, prep_right, seasonal_cumulative
from .run_forecast import FEATURES, log, make_train
from .series import HERE, KEY, prepare

DEFAULT = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_child_samples=200,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0)
# parameters whose sensible value grows with the number of training rows
ROW_SCALED = ("min_child_samples", "min_data_per_group")


def km_mae(model, X, y_rate, gap, w, cap) -> float:
    """Row-weighted MAE in km (error = gap * rate error; w = re-weighting of sampled targets)."""
    return float(np.average(np.abs(np.clip(model.predict(X), 0, cap) - y_rate) * gap, weights=w))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--rows", type=int, default=600_000, help="training rows per trial")
    ap.add_argument("--frac", type=float, default=1.0, help="share of training vehicles (smoke tests)")
    ap.add_argument("--cutoff", default="2025-07-01")
    ap.add_argument("--val_end", default="2025-11-11")
    ap.add_argument("--max_targets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--drop", default="")
    ap.add_argument("--tag", default="hpo")
    ap.add_argument("--skip_heldout", action="store_true", help="skip the final default-vs-tuned fit")
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)      # lightgbm eval_set deprecation spam
    warnings.simplefilter("ignore", DeprecationWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feats = [f for f in FEATURES if f not in args.drop.split(",")]
    rng = np.random.default_rng(args.seed)
    out_dir = HERE / "results" / "hpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # same vehicle split as run_forecast (seed 0 => identical held-out vehicles)
    log("loading + cleaning")
    all_, ok = prepare(HERE / "data" / "data_30_percent_sample.parquet")
    c_val, v_end = day_of(args.cutoff), day_of(args.val_end)
    vids = np.sort(all_[KEY].unique())
    rng.shuffle(vids)
    n_val = int(0.2 * len(vids))
    val_v, rest = vids[:n_val], vids[n_val:]
    n_inner = int(0.1 * len(rest))
    inner_v = rest[:n_inner]
    train_v = rest[n_inner:int(n_inner + (len(rest) - n_inner) * args.frac)]
    F = seasonal_cumulative(ok[ok[KEY].isin(np.concatenate([train_v, inner_v]))], c_val)
    cutoffs = list(range(day_of("2023-03-01"), c_val - 7, 30))
    log("building samples")
    tr = make_train(ok, all_, train_v, cutoffs, c_val, F, args, rng)
    inner_all = make_train(ok, all_, inner_v, cutoffs, c_val, F, args, rng)
    # search slice: full-horizon recent cutoffs (see module docstring)
    inner = inner_all[(inner_all.cutoff >= c_val - 365) & (inner_all.cutoff <= c_val - 150)]
    sub = tr.sample(min(args.rows, len(tr)), random_state=args.seed)
    cap = float(tr.y_rate.quantile(0.999))
    scale = len(tr) / len(sub)
    log(f"HPO rows {len(sub):,} / train {len(tr):,} (x{scale:.1f}); inner slice {len(inner):,} "
        f"rows, gap median {inner.gap.median():.0f} d")

    def objective(trial: optuna.Trial) -> float:
        p = dict(objective="l1", learning_rate=0.1, subsample_freq=1, n_jobs=args.n_jobs,
                 verbose=-1, random_state=args.seed,
                 num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 20, 1000, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                 min_split_gain=trial.suggest_float("min_split_gain", 1e-3, 1.0, log=True),
                 # brnc_pool_code carries ~40% of the gain: regularise high-cardinality splits
                 cat_smooth=trial.suggest_float("cat_smooth", 1, 200, log=True),
                 cat_l2=trial.suggest_float("cat_l2", 1, 100, log=True),
                 max_cat_threshold=trial.suggest_int("max_cat_threshold", 8, 128, log=True),
                 min_data_per_group=trial.suggest_int("min_data_per_group", 20, 500, log=True))
        m = lgb.LGBMRegressor(n_estimators=2000, **p)
        m.fit(sub[feats], sub.y_rate, sample_weight=sub.gap * sub.w_tgt,
              eval_set=[(inner[feats], inner.y_rate)], eval_sample_weight=[inner.gap * inner.w_tgt],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        trial.set_user_attr("best_iter", int(m.best_iteration_))
        return km_mae(m, inner[feats], inner.y_rate, inner.gap, inner.w_tgt, cap)

    study = optuna.create_study(direction="minimize", study_name=args.tag,
                                storage=f"sqlite:///{(out_dir / (args.tag + '.db')).as_posix()}",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True))
    if len(study.trials) == 0:           # default params as the first trial (not again on resume)
        study.enqueue_trial({"num_leaves": 63, "min_child_samples": 200, "colsample_bytree": 0.8,
                             "subsample": 0.8, "reg_lambda": 5.0, "min_split_gain": 1e-3,
                             "cat_smooth": 10.0, "cat_l2": 10.0, "max_cat_threshold": 32,
                             "min_data_per_group": 100})
    def progress(st, t):
        if t.value is not None:
            log(f"trial {t.number}: {t.value:.1f} km (best {st.best_value:.1f}), "
                f"{t.user_attrs.get('best_iter')} trees")
    study.optimize(objective, timeout=args.minutes * 60, gc_after_trial=True, callbacks=[progress])
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    study.trials_dataframe().to_csv(out_dir / f"{args.tag}_trials.csv", index=False)
    best = study.best_params
    # tuned on `sub`; the full fit has `scale` x more rows => scale the row-count parameters
    final = {k: (int(round(v * scale)) if k in ROW_SCALED else v) for k, v in best.items()}
    default_inner = done[0].value if done else float("nan")
    json.dump({"best_params": best, "final_params": final, "row_scale": scale,
               "inner_km_mae": study.best_value, "default_inner_km_mae": default_inner,
               "n_trials": len(done)}, open(out_dir / f"{args.tag}_best.json", "w"), indent=2)
    log(f"{len(done)} trials; best inner MAE {study.best_value:.1f} km (default {default_inner:.1f}); "
        f"final params {final}")
    if args.skip_heldout:
        return

    # ---- final comparison on the held-out vehicles (full training rows, lr 0.05, clipped)
    right, first = prep_right(ok, val_v)
    val = build_samples(right, first, all_[all_[KEY].isin(val_v)], c_val, v_end - c_val, F)
    res = {}
    for name, extra in [("default", {}), ("tuned", final)]:
        p = {**DEFAULT, **extra, "n_jobs": args.n_jobs, "verbose": -1, "random_state": args.seed}
        m = lgb.LGBMRegressor(n_estimators=4000, **p)
        m.fit(tr[feats], tr.y_rate, sample_weight=tr.gap * tr.w_tgt,
              eval_set=[(inner_all[feats], inner_all.y_rate)],
              eval_sample_weight=[inner_all.gap * inner_all.w_tgt],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        err = val.km0 + np.clip(m.predict(val[feats]), 0, cap) * val.gap - val.km_t
        res[name] = {"MAE": float(err.abs().mean()), "MedAE": float(err.abs().median()),
                     "bias": float(err.mean()), "trees": int(m.best_iteration_)}
        log(f"{name}: {res[name]}")
    json.dump(res, open(out_dir / f"{args.tag}_heldout.json", "w"), indent=2)
    print(pd.DataFrame(res).T.round(1))


if __name__ == "__main__":
    main()
