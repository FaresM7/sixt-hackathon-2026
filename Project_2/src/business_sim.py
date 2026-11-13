"""Business value simulations on the saved validation predictions (no new model training).

1) "Ask for readings where it matters": at the cutoff, SIXT requests a fresh odometer reading from
   k% of the held-out cars. For those cars the starting point becomes the true km at the cutoff
   (interpolated between the last reading before and the first reading after it) and the model's
   predicted km/day is applied from the cutoff on. Which cars to ask: the ones with the WIDEST 80%
   band (our uncertainty), vs the cars whose last reading is oldest, vs random cars.
2) Fleet-level error: errors of single cars partly cancel when km are summed per branch pool /
   region / fleet, which is what valuation and defleet planning use.

Needs results/val_preds_tuned.parquet (python -m src.run_forecast ... --bands --tag _tuned).
Usage: python -m src.business_sim
"""
import numpy as np
import pandas as pd

from .series import HERE, KEY, prepare


def true_km_at(ok: pd.DataFrame, day: int) -> pd.Series:
    """True odometer per vehicle at `day`: interpolation between the last clean reading on/before
    and the first clean reading after it (NaN if the car has no reading on one side)."""
    before = ok[ok.day <= day].groupby(KEY).last()[["day", "km"]]
    after = ok[ok.day > day].groupby(KEY).first()[["day", "km"]]
    b = before.join(after, lsuffix="0", rsuffix="1", how="inner")
    frac = (day - b.day0) / (b.day1 - b.day0)
    return b.km0 + (b.km1 - b.km0).clip(lower=0) * frac


def main() -> None:
    res = HERE / "results"
    v = pd.read_parquet(res / "val_preds_tuned.parquet")
    c = int(v.cutoff.iloc[0])
    _, ok = prepare(HERE / "data" / "data_30_percent_sample.parquet")
    ok = ok[ok[KEY].isin(v[KEY].unique())]
    car = v.groupby(KEY).agg(stale=("stale", "first"), hi=("hi", "mean"), lo=("lo", "mean"))
    car["band"] = car.hi - car.lo
    rate = (v.m_lgbm - v.km0) / v.gap
    err_now = (v.m_lgbm - v.km_t).abs()
    total = err_now.sum()
    rng = np.random.default_rng(0)
    cars = car.index.to_numpy()
    rows = []
    for offset, label in [(0, "reading on the cutoff day"), (60, "check-in reading 60 days after the cutoff")]:
        r = c + offset
        km_r = true_km_at(ok, r).reindex(car.index)
        km_r_row = v[KEY].map(km_r)
        after = v.day_t > r
        pred_new = km_r_row + rate * (v.day_t - r)                       # restart from the true km at r
        err_new = (pred_new - v.km_t).abs().where(after & km_r_row.notna(), err_now)
        gain_car = (err_now - err_new).groupby(v[KEY]).sum()             # error removed per car asked
        for k in (0.05, 0.10, 0.20):
            n = round(k * len(cars))
            picks = {"widest 80% band (ours)": car.band.nlargest(n).index,
                     "oldest last reading": car.stale.nlargest(n).index}
            for name, idx in picks.items():
                rows.append({"scenario": label, "cars_asked": f"{k:.0%}", "strategy": name,
                             "fleet_error_reduction": gain_car.reindex(idx).fillna(0).sum() / total})
            rnd = [gain_car.reindex(rng.choice(cars, n, replace=False)).fillna(0).sum() / total
                   for _ in range(50)]
            rows.append({"scenario": label, "cars_asked": f"{k:.0%}", "strategy": "random cars (mean of 50)",
                         "fleet_error_reduction": float(np.mean(rnd))})
        rows.append({"scenario": label, "cars_asked": "100%", "strategy": "all cars",
                     "fleet_error_reduction": gain_car.sum() / total})
    voi = pd.DataFrame(rows)
    print("=== Share of total forecast error removed by requesting readings from k% of cars ===")
    print(voi.pivot_table(index=["scenario", "strategy"], columns="cars_asked",
                          values="fleet_error_reduction").round(3).to_string())
    voi.to_csv(res / "business_reading_requests.csv", index=False)

    # ---- fleet-level aggregation: relative error of summed km driven since the last reading
    v["true_d"], v["pred_d"] = v.km_t - v.km0, v.m_lgbm - v.km0
    rows = [{"level": "single reading (WAPE)", "groups": len(v),
             "relative_error": (v.pred_d - v.true_d).abs().sum() / v.true_d.abs().sum()}]
    for name, col in [("branch pool", "brnc_pool_code"), ("region", "brnc_region_code")]:
        g = v.groupby(col, observed=True)[["pred_d", "true_d"]].sum()
        g = g[g.true_d > 0]
        rows.append({"level": f"sum per {name}", "groups": len(g),
                     "relative_error": (g.pred_d - g.true_d).abs().sum() / g.true_d.sum()})
    rows.append({"level": "whole held-out fleet", "groups": 1,
                 "relative_error": abs(v.pred_d.sum() - v.true_d.sum()) / v.true_d.sum()})
    agg = pd.DataFrame(rows)
    print("\n=== Error relative to km driven, by aggregation level ===")
    print(agg.round(3).to_string(index=False))
    agg.to_csv(res / "business_fleet_aggregation.csv", index=False)


if __name__ == "__main__":
    main()
