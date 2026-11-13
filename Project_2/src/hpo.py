"""Hyper-parameter search for the LightGBM rate model (Optuna/TPE, time-capped, resumable).

Search objective: MAE in km (weighted L1 on the rate, weight = gap) on a vehicle-disjoint inner
validation slice, using a subsample of training rows and learning rate 0.1 for speed. The held-out
validation vehicles are only touched at the end: best params vs. default params, full data.

Usage: python -m src.hpo [--minutes 35] [--rows 600000] [--drop f1,f2]
Resume: rerun with the same --tag (study stored in results/hpo/<tag>.db).
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from .features import build_samples, day_of, prep_right, seasonal_cumulative
from .run_forecast import FEATURES, log, make_train
from .series import HERE, KEY, prepare

DEFAULT = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_child_samples=200,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0)


def km_mae(model, X, y_rate, gap) -> float:
    """MAE in km of km0 + rate*gap given rate predictions (error = gap * rate error)."""
    return float(np.mean(np.abs(model.predict(X) - y_rate) * gap))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--rows", type=int, default=600_000, help="training rows per trial")
    ap.add_argument("--cutoff", default="2025-07-01")
    ap.add_argument("--val_end", default="2025-11-11")
    ap.add_argument("--max_targets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default="")
    ap.add_argument("--tag", default="hpo")
    args = ap.parse_args()
    feats = [f for f in FEATURES if f not in args.drop.split(",")]
    rng = np.random.default_rng(args.seed)
    out_dir = HERE / "results" / "hpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # same vehicle split as run_forecast (seed 0 => identical held-out vehicles)
    all_, ok = prepare(HERE / "data" / "data_30_percent_sample.parquet")
    c_val, v_end = day_of(args.cutoff), day_of(args.val_end)
    vids = np.sort(all_[KEY].unique())
    rng.shuffle(vids)
    n_val = int(0.2 * len(vids))
    val_v, rest = vids[:n_val], vids[n_val:]
    inner_v, train_v = rest[: int(0.1 * len(rest))], rest[int(0.1 * len(rest)):]
    F = seasonal_cumulative(ok[ok[KEY].isin(rest)], c_val)
    cutoffs = list(range(day_of("2023-03-01"), c_val - 7, 30))
    tr = make_train(ok, all_, train_v, cutoffs, c_val, F, args, rng)
    inner = make_train(ok, all_, inner_v, cutoffs, c_val, F, args, rng)
    inner = inner[inner.cutoff >= c_val - 150]          # recent cutoffs: closest to the real task
    sub = tr.sample(min(args.rows, len(tr)), random_state=args.seed)
    log(f"HPO rows {len(sub):,} / train {len(tr):,}; inner {len(inner):,}")

    def objective(trial: optuna.Trial) -> float:
        p = dict(objective="l1", learning_rate=0.1, subsample_freq=1, n_jobs=16, verbose=-1,
                 random_state=args.seed,
                 num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 50, 1000, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                 cat_smooth=trial.suggest_float("cat_smooth", 5, 200, log=True),
                 cat_l2=trial.suggest_float("cat_l2", 1, 100, log=True))
        m = lgb.LGBMRegressor(n_estimators=1500, **p)
        m.fit(sub[feats], sub.y_rate, sample_weight=sub.gap,
              eval_set=[(inner[feats], inner.y_rate)], eval_sample_weight=[inner.gap],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        trial.set_user_attr("best_iter", int(m.best_iteration_))
        return km_mae(m, inner[feats], inner.y_rate, inner.gap)

    study = optuna.create_study(direction="minimize", study_name=args.tag,
                                storage=f"sqlite:///{(out_dir / (args.tag + '.db')).as_posix()}",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.enqueue_trial({"num_leaves": 63, "min_child_samples": 200, "colsample_bytree": 0.8,
                         "subsample": 0.8, "reg_lambda": 5.0, "cat_smooth": 10.0, "cat_l2": 10.0})
    study.optimize(objective, timeout=args.minutes * 60, gc_after_trial=True)
    study.trials_dataframe().to_csv(out_dir / f"{args.tag}_trials.csv", index=False)
    best = study.best_params
    json.dump({"best_params": best, "inner_km_mae": study.best_value,
               "n_trials": len(study.trials)}, open(out_dir / f"{args.tag}_best.json", "w"), indent=2)
    log(f"{len(study.trials)} trials; best inner MAE {study.best_value:.1f} km; params {best}")

    # ---- final comparison on the held-out vehicles (full training rows, lr 0.05)
    right, first = prep_right(ok, val_v)
    val = build_samples(right, first, all_[all_[KEY].isin(val_v)], c_val, v_end - c_val, F)
    res = {}
    for name, extra in [("default", {}), ("tuned", best)]:
        p = {**DEFAULT, **extra, "n_jobs": 16, "verbose": -1, "random_state": args.seed}
        m = lgb.LGBMRegressor(n_estimators=3000, **p)
        m.fit(tr[feats], tr.y_rate, sample_weight=tr.gap, eval_set=[(inner[feats], inner.y_rate)],
              eval_sample_weight=[inner.gap], callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = val.km0 + m.predict(val[feats]) * val.gap
        res[name] = {"MAE": float((pred - val.km_t).abs().mean()),
                     "MedAE": float((pred - val.km_t).abs().median()),
                     "trees": int(m.best_iteration_)}
        log(f"{name}: {res[name]}")
    json.dump(res, open(out_dir / f"{args.tag}_heldout.json", "w"), indent=2)
    print(pd.DataFrame(res).T.round(1))


if __name__ == "__main__":
    main()
