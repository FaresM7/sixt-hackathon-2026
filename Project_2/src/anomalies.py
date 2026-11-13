"""Erroneous mileage detection (task B): rules + two-sided score + forecast-based real-time score.

Three detectors, all label-free:
- RULES (physical, explainable): mileage drops vs the previous reading, impossible km/day since
  the previous reading, absurd values, odometer reset to ~0.
- TWO-SIDED score (historical reports, readings before AND after are known): an odometer only goes
  up, so a reading must lie between the median of its 3 previous and the median of its 3 next
  readings. Score = km outside that bracket. No model needed, very precise for past reports.
- FORWARD score (real-time: a new report, nothing after it yet): the task-A LightGBM forecast made
  from the history before the report; score = |reported - expected| / typical error at that gap.

Evaluation (no labels exist): inject known errors into 1% of the clean daily readings of the
held-out vehicles (dropped digit, extra digit, swapped digits, rollback, upward offset, stuck
odometer, miles reported as km). For each detector, the threshold is set to an ALARM BUDGET
(share of non-injected readings flagged, e.g. 1%) and we report recall per error type.
Precision is reported as a lower bound (every flag on a non-injected reading counts as false,
although many of those are real errors) next to the flag rate on the real data.

Typo auto-correction: for flagged readings, try digit edits and the miles->km factor; suggest
the candidate that falls inside the bracket closest to the interpolated expectation.

Usage: python -m src.anomalies [--no_forward] [--frac 0.3]
Outputs: results/anomaly_eval.csv, results/anomaly_flags_top.csv, results/plots/anomaly_examples.png
"""
import argparse
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .features import build_samples, day_of, prep_right, seasonal_cumulative
from .run_forecast import FEATURES, log, make_train
from .series import HERE, KEY, prepare

TYPES = ["drop_digit", "extra_digit", "swap_digits", "rollback", "offset_up", "stuck", "miles_as_km"]
BUDGETS = [0.005, 0.01, 0.02]
MILE = 1.609344
NB = 5            # neighbours per side for the robust bracket
AGREE_KM = 300.0  # a reading consistent with one side (+ AGREE_RATE km/day) is not blamed
AGREE_RATE = 150.0
PASS1_KM = 500.0  # pass-1 suspects are excluded from the pass-2 brackets


# ----------------------------------------------------------------------------- detectors
def neighbours(s: pd.DataFrame) -> pd.DataFrame:
    """Robust bracket per reading, in two passes. Pass 1: bracket = median of the NB previous /
    NB next readings; readings > PASS1_KM outside it are suspects. Pass 2: the bracket is rebuilt
    from non-suspect neighbours only, so a block of several bad readings in a row neither hides
    itself nor gets the correct reading next to it blamed. Adds lo, hi, two_sided, expected."""
    s = s.sort_values([KEY, "day"]).reset_index(drop=True)
    s = _bracket(s, s.km)
    s["two_sided"] = two_sided_score(s)
    s = _bracket(s, s.km.where(s.two_sided <= PASS1_KM))
    s["two_sided"] = two_sided_score(s)
    return s


def _bracket(s: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    g = s.assign(ref=ref).groupby(KEY)
    # 5 neighbours per side: a block of up to 2 bad readings in a row cannot shift the median
    prev = np.column_stack([g.ref.shift(k) for k in range(1, NB + 1)])
    nxt = np.column_stack([g.ref.shift(-k) for k in range(1, NB + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s["lo"], s["hi"] = np.nanmedian(prev, axis=1), np.nanmedian(nxt, axis=1)
    mid = (NB + 1) // 2                                         # the median sits ~mid readings away
    s["span_p"] = (s.day - g.day.shift(mid)).fillna(0)
    s["span_n"] = (g.day.shift(-mid) - s.day).fillna(0)
    s["km_p"], s["day_p"] = g.km.shift(1), g.day.shift(1)
    s["km_n"], s["day_n"] = g.km.shift(-1), g.day.shift(-1)
    frac = (s.day - s.day_p) / (s.day_n - s.day_p)
    s["expected"] = (s.km_p + (s.km_n - s.km_p).clip(lower=0) * frac).clip(s.lo, s.hi)
    return s


def two_sided_score(s: pd.DataFrame) -> np.ndarray:
    """km by which a reading disagrees with both its previous and its next readings."""
    # outside the bracket, unless the reading agrees (within AGREE_KM) with one side: a correct
    # reading next to a block of bad ones matches the readings on its other side => not blamed.
    lo, hi, km = s.lo.to_numpy(), s.hi.to_numpy(), s.km.to_numpy()
    out = np.nan_to_num(np.fmax(np.fmax(lo - km, km - hi), 0.0))      # outside the bracket
    # agreeing = consistent with normal driving between the reading and that side's median
    # (which sits ~3 readings away): within AGREE_KM + AGREE_RATE km/day of it, right direction
    dp, dn = s.span_p.to_numpy(), s.span_n.to_numpy()
    agree_p = (km - lo >= -AGREE_KM) & (km - lo <= AGREE_KM + AGREE_RATE * dp)
    agree_n = (hi - km >= -AGREE_KM) & (hi - km <= AGREE_KM + AGREE_RATE * dn)
    agrees = agree_p | agree_n
    return np.where(agrees, 0.0, out)


def rule_flags(s: pd.DataFrame, rate_cap: float) -> pd.Series:
    """Reason string of the first physical rule that fires ('' if none). The drop / rate rules
    only fire on the reading that is itself outside its bracket, so the correct reading right
    after a bad one is not blamed (needs column `two_sided`)."""
    culprit = s.two_sided > 0
    rate = (s.km - s.km_p) / (s.day - s.day_p)
    reason = pd.Series("", index=s.index, dtype=object)
    for name, hit in [
        ("absurd value (>=900k or >3x max mileage)", (s.km >= 900_000) | (s.km > 3 * s.vhcl_max_mileage)),
        ("reset to ~0 after real mileage", (s.km < 10) & (s.km_p > 1_000)),
        ("mileage decreased vs previous reading", (s.km < s.km_p - 100) & culprit),
        (f"implied usage > {rate_cap:.0f} km/day since previous reading", (rate > rate_cap) & culprit),
    ]:
        reason = reason.where(reason != "", np.where(hit.fillna(False), name, ""))
    return reason


def interval_rate_cap(s: pd.DataFrame) -> float:
    """p99.9 of km/day between consecutive clean readings (>= 1 day apart), at least 1,000."""
    o = s[s.ok]
    g = o.groupby(KEY)
    r = (g.km.diff() / g.day.diff()).replace([np.inf, -np.inf], np.nan).dropna()
    return float(max(r.quantile(0.999), 1_000.0))


# ----------------------------------------------------------------------------- injection
def _digits_edit(km: int, kind: str, rng) -> float:
    t = str(int(km))
    if kind == "drop_digit":
        p = rng.integers(0, len(t))
        return float(t[:p] + t[p + 1:])
    if kind == "extra_digit":
        p = rng.integers(1, len(t) + 1)
        return float(t[:p] + str(rng.integers(0, 10)) + t[p:])
    pairs = [i for i in range(len(t) - 1) if t[i] != t[i + 1]]
    if not pairs:
        return np.nan
    i = pairs[rng.integers(0, len(pairs))]
    return float(t[:i] + t[i + 1] + t[i] + t[i + 2:])


def inject(s: pd.DataFrame, vehicles, rate: float, rng) -> pd.DataFrame:
    """Corrupt `rate` of the clean, interior readings (km >= 1000) of `vehicles`, never two in a row."""
    s = s.copy()
    s["km"] = s.km.astype(float)
    s["label"], s["itype"], s["km_orig"] = 0, "", s.km
    g = s.groupby(KEY)
    interior = g.cumcount().gt(0) & g.cumcount(ascending=False).gt(0)
    cand = s.index[s[KEY].isin(vehicles) & s.ok & interior & (s.km >= 1_000)].to_numpy()
    pick = np.sort(rng.choice(cand, size=int(rate * len(cand)), replace=False))
    pick = pick[np.r_[True, np.diff(pick) > 1]]                 # no adjacent injections
    types = rng.choice(TYPES, size=len(pick))
    km = s.km.to_numpy().copy()
    km_prev = g.km.shift(1).to_numpy()
    new = np.full(len(pick), np.nan)
    for j, (i, t) in enumerate(zip(pick, types)):
        if t in ("drop_digit", "extra_digit", "swap_digits"):
            new[j] = _digits_edit(km[i], t, rng)
        elif t == "rollback":
            new[j] = max(km[i] - rng.uniform(1_000, 20_000), 0)
        elif t == "offset_up":
            new[j] = km[i] + rng.uniform(2_000, 20_000)
        elif t == "stuck":
            new[j] = km_prev[i] if km[i] - km_prev[i] > 100 else np.nan
        elif t == "miles_as_km":
            new[j] = round(km[i] / MILE)
    keep = ~np.isnan(new) & (np.abs(new - km[pick]) > 0)
    pick, types, new = pick[keep], types[keep], new[keep]
    s.loc[pick, "km"] = new
    s.loc[pick, "label"] = 1
    s.loc[pick, "itype"] = types
    return s


# ----------------------------------------------------------------------------- typo correction
def suggest_fix(km: float, lo: float, hi: float, expected: float) -> float:
    """Closest-to-expected candidate (digit edits, miles->km) inside [lo, hi]; NaN if none."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
        return np.nan
    t = str(int(km))
    cands = {round(km * MILE)}
    for p in range(len(t) + 1):                                   # a digit was dropped
        cands.update(int(t[:p] + str(d) + t[p:]) for d in range(10))
    for p in range(len(t)):                                       # a digit was added
        if len(t) > 1:
            cands.add(int(t[:p] + t[p + 1:]))
    for i in range(len(t) - 1):                                   # two digits swapped
        cands.add(int(t[:i] + t[i + 1] + t[i] + t[i + 2:]))
    inside = [c for c in cands if lo <= c <= hi and c != km]
    if not inside:
        return np.nan
    ref = expected if np.isfinite(expected) else (lo + hi) / 2
    return float(min(inside, key=lambda c: abs(c - ref)))


# ----------------------------------------------------------------------------- evaluation helpers
def budget_table(d: pd.DataFrame, scores: dict, rules: pd.Series | None) -> pd.DataFrame:
    """Recall per error type at alarm budgets (share of non-injected readings flagged)."""
    rows = []
    neg = d.label.to_numpy() == 0
    for name, sc in scores.items():
        sc = np.asarray(sc, dtype=float)
        for b in BUDGETS:
            thr = np.quantile(sc[neg], 1 - b)
            flag = sc > max(thr, 0)
            rows.append(_row(d, name, f"{b:.1%}", flag, float(thr)))
        rows[-1]["AP"] = rows[-2]["AP"] = rows[-3]["AP"] = average_precision_score(d.label, sc)
    if rules is not None:
        flag = (rules != "").to_numpy()
        rows.append(_row(d, "rules", "fixed", flag, np.nan))
        for b in BUDGETS:      # rules first, fill the rest of the budget with the two-sided score
            sc = np.asarray(scores["two_sided"], dtype=float)
            room = max(b - flag[neg].mean(), 0)
            thr = np.quantile(sc[neg & ~flag], 1 - room) if room > 0 else np.inf
            rows.append(_row(d, "rules+two_sided", f"{b:.1%}", flag | (sc > max(thr, 0)), float(thr)))
    return pd.DataFrame(rows)


def _row(d, name, budget, flag, thr) -> dict:
    lab = d.label.to_numpy() == 1
    r = {"detector": name, "budget": budget, "threshold": thr,
         "flag_rate_real": flag[~lab].mean(), "recall_all": flag[lab].mean(),
         "precision_lb": flag[lab].sum() / max(flag.sum(), 1)}
    if "km_orig" in d:                      # errors that matter: >= 1,000 km off
        big = lab & (np.abs(d.km - d.km_orig).to_numpy() >= 1_000)
        r["recall_err_ge_1000km"] = flag[big].mean() if big.any() else np.nan
    for t in TYPES:
        m = (d.itype == t).to_numpy()
        r[f"rec_{t}"] = flag[m].mean() if m.any() else np.nan
    return r


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.01, help="share of clean readings corrupted")
    ap.add_argument("--frac", type=float, default=0.3, help="training vehicles for the forward model")
    ap.add_argument("--no_forward", action="store_true", help="skip the forecast-based score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cutoff", default="2025-07-01")
    ap.add_argument("--val_end", default="2025-11-11")
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    rng = np.random.default_rng(args.seed)
    res = HERE / "results"
    (res / "plots").mkdir(parents=True, exist_ok=True)

    log("loading + cleaning")
    all_, ok = prepare(HERE / "data" / "data_30_percent_sample.parquet")
    vids = np.sort(all_[KEY].unique())
    rng.shuffle(vids)                                   # same split as run_forecast (seed 0)
    n_val = int(0.2 * len(vids))
    val_v, rest = vids[:n_val], vids[n_val:]
    cap = interval_rate_cap(all_)
    log(f"rate cap {cap:.0f} km/day")

    # ---- 1) real data: flags on ALL readings (no injection)
    real = neighbours(all_)
    real["rule"] = rule_flags(real, cap)

    # ---- 2) injection on the held-out vehicles (their whole history)
    inj = inject(all_, val_v, args.rate, np.random.default_rng(args.seed + 1))
    ev = neighbours(inj)
    ev = ev[ev[KEY].isin(val_v)].reset_index(drop=True)
    ev["rule"] = rule_flags(ev, cap)
    log(f"injected {ev.label.sum():,} errors into {len(ev):,} readings of {len(val_v):,} held-out vehicles")
    # baseline = starter.py's flags: mileage goes down, or > 1000 km/day since the previous reading
    st_rate = (ev.km - ev.km_p) / (ev.day - ev.day_p)
    st_flag = ((ev.km < ev.km_p) | (st_rate > 1000)).to_numpy()
    tab = pd.concat([pd.DataFrame([_row(ev, "starter_rules (baseline)", "fixed", st_flag, np.nan)]),
                     budget_table(ev, {"two_sided": ev.two_sided}, ev.rule)]).assign(scope="history (two-sided)")

    # ---- 3) typo auto-correction on injected digit / unit errors that the two-sided score catches
    thr1 = float(tab[(tab.detector == "two_sided") & (tab.budget == "1.0%")].threshold.iloc[0])
    fixable = ev[(ev.label == 1) & ev.itype.isin(["drop_digit", "extra_digit", "swap_digits", "miles_as_km"])
                 & (ev.two_sided > thr1)]
    fix = np.array([suggest_fix(r.km, r.lo, r.hi, r.expected) for r in fixable.itertuples()])
    exact = np.abs(fix - fixable.km_orig.to_numpy()) <= 1
    close = np.abs(fix - fixable.km_orig.to_numpy()) <= 100
    corr = (pd.DataFrame({"itype": fixable.itype, "suggested": ~np.isnan(fix), "exact": exact, "within_100km": close})
            .groupby("itype").mean().round(3))
    corr["n"] = fixable.groupby("itype").size()
    log(f"typo auto-correction on flagged injected typos:\n{corr.to_string()}")
    corr.to_csv(res / "anomaly_correction.csv")

    # ---- 4) forward (real-time) score with the task-A model, on the validation window
    if not args.no_forward:
        c_val, v_end = day_of(args.cutoff), day_of(args.val_end)
        n_inner = int(0.1 * len(rest))
        inner_v = rest[:n_inner]
        train_v = rest[n_inner:int(n_inner + (len(rest) - n_inner) * args.frac)]
        F = seasonal_cumulative(ok[ok[KEY].isin(rest)], c_val)
        cutoffs = list(range(day_of("2023-03-01"), c_val - 7, 30))
        a = argparse.Namespace(max_targets=8)
        tr = make_train(ok, all_, train_v, cutoffs, c_val, F, a, rng)
        inner = make_train(ok, all_, inner_v, cutoffs, c_val, F, a, rng)
        params = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0)
        hp = res / "hpo" / "hpo_best.json"
        if hp.exists():
            params.update(json.load(open(hp))["final_params"])
        log(f"training forward model on {len(tr):,} rows")
        m = lgb.LGBMRegressor(n_estimators=700, n_jobs=16, verbose=-1, random_state=args.seed, **params)
        m.fit(tr[FEATURES], tr.y_rate, sample_weight=tr.gap * tr.w_tgt)
        rcap = float(tr.y_rate.quantile(0.999))
        # typical absolute error per gap bucket, from the inner slice (no held-out data, no labels)
        edges = [14, 30, 60, 90, 10_000]
        res_in = np.abs(np.clip(m.predict(inner[FEATURES]), 0, rcap) - inner.y_rate) * inner.gap
        b_in = np.digitize(inner.gap, edges, right=True)
        scale = pd.Series(res_in.to_numpy()).groupby(b_in).median().reindex(range(len(edges))).ffill().to_numpy()
        # targets = (possibly corrupted) readings of held-out vehicles in the window; clean history
        right, first = prep_right(ok, val_v)
        tgt = inj[inj[KEY].isin(val_v)][[KEY, "day", "km", "ok", "label", "itype", "km_orig"]]
        win = build_samples(right, first, tgt[[KEY, "day", "km", "ok"]], c_val, v_end - c_val, F)
        win = win.merge(tgt.rename(columns={"day": "day_t"})[[KEY, "day_t", "label", "itype", "km_orig"]],
                        on=[KEY, "day_t"], how="left")
        pred = win.km0 + np.clip(m.predict(win[FEATURES]), 0, rcap) * win.gap
        win["forward"] = (win.km_t - pred).abs() / scale[np.digitize(win.gap, edges, right=True)]
        win["km"] = win.km_t
        w2 = win.merge(ev[[KEY, "day", "two_sided"]].rename(columns={"day": "day_t"}), on=[KEY, "day_t"])
        log(f"forward eval: {len(w2):,} window readings, {int(w2.label.sum())} injected")
        tab = pd.concat([tab, budget_table(w2, {"forward_model": w2.forward, "two_sided": w2.two_sided}, None)
                         .assign(scope="window after cutoff (real-time vs two-sided)")])

    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = ["scope", "detector", "budget", "flag_rate_real", "recall_all", "precision_lb", "AP"] + \
           [f"rec_{t}" for t in TYPES]
    print("\n=== Injection evaluation (recall per error type at an alarm budget) ===")
    print(tab[show].round(3).to_string(index=False))
    tab.to_csv(res / "anomaly_eval.csv", index=False)

    # ---- 5) real flags: rules (certain) + two-sided above the 1% budget threshold (likely)
    real["tier"] = np.where(real.rule != "", "certain", np.where(real.two_sided > thr1, "likely", ""))
    real["reason"] = np.where(real.rule != "", real.rule,
                              "outside the range of the readings before/after by " + real.two_sided.round(0).astype(int).astype(str) + " km")
    flagged = real[real.tier != ""].copy()
    flagged["suggested_km"] = [suggest_fix(r.km, r.lo, r.hi, r.expected) for r in flagged.itertuples()]
    log(f"real data: {len(flagged):,} flagged readings ({len(flagged) / len(real):.2%}); "
        f"tiers {flagged.tier.value_counts().to_dict()}; with a suggested correction: "
        f"{flagged.suggested_km.notna().mean():.1%}")
    top = flagged.sort_values("two_sided", ascending=False)
    cols = [KEY, "day", "km", "km_p", "km_n", "lo", "hi", "two_sided", "tier", "reason", "suggested_km"]
    top[cols].head(500).to_csv(res / "anomaly_flags_top.csv", index=False)
    plot_examples(real, top, res / "plots" / "anomaly_examples.png")
    log("done")


def plot_examples(real: pd.DataFrame, top: pd.DataFrame, path) -> None:
    """6 real flagged readings (mixed reasons), each with +-120 days of the car's history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = pd.concat([top[top.tier == "likely"].iloc[[0, 20, 200]],
                       top[top.reason.str.startswith("mileage decreased")].iloc[[0, 30]],
                       top[top.suggested_km.notna() & (top.tier == "likely")].iloc[[5, 50]]])
    picks = picks.drop_duplicates(subset=[KEY]).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, r in zip(axes.ravel(), picks.itertuples()):
        h = real[(real[KEY] == getattr(r, KEY)) & (real.day.between(r.day - 120, r.day + 120))]
        ax.plot(h.day - r.day, h.km, ".-", color="0.5", lw=1, label="reported")
        ax.plot(0, r.km, "o", color="crimson", ms=9, label="flagged")
        if np.isfinite(r.suggested_km):
            ax.plot(0, r.suggested_km, "x", color="green", ms=10, mew=2, label="suggested fix")
        ax.set_title(f"{r.tier}: {r.reason[:60]}", fontsize=8)
        ax.set_xlabel("days relative to flagged reading")
        ax.set_ylabel("km")
        ax.legend(fontsize=7)
    fig.suptitle("Flagged real mileage reports (rules = certain, two-sided score = likely)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
