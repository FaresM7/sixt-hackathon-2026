# PLAN: Project 2, Mileage Nowcast / Forecast / Anomaly Flagging

> Working plan, kept for transparency: Part I was written before seeing the data, Part II (sections 8-15) records the decisions made during the day. Final approach and results: `SUBMISSION.md`.

## 0. Core idea (one sentence for the pitch)
Model each vehicle's odometer as `last known reading + (usage rate) x (elapsed days)`, learn the usage rate with a gradient-boosted model plus quantile bands, and use the same expected value + band to flag reports that fall outside it, backed by physical-plausibility rules.

## 1. Data understanding (first 45 min)
1. Load, parse dates, sort by (vehicle, date, mileage). Report: rows, vehicles, date range, readings per vehicle, gap distribution between readings (days), share of long-term vs short-term, missingness.
2. **Verify the row semantics** (see CLAUDE.md hypothesis: OnRent row = handover km, NotOnRent row = return km). Check `return_km[i] == handover_km[i+1]` on the full data. Ask owners if unclear. This decides how we build the "observation table".
3. Build a clean **observation table**: one row per (vehicle, date) odometer observation with a type (`handover` / `return` / `snapshot`) and the rental it belongs to. Dedupe exact duplicates. Resolve same-day conflicts by type, not by max.
4. Look at the shape of the rate distribution (km/day), per vehicle_level, per rental_product, per month. Identify obvious bad reports (decreases, >1000 km/day, mileage > max mileage).

## 2. Evaluation design (decide BEFORE modelling, then freeze)
**Splits** (both, report both):
- *Time split*: train on observations before cutoff C (e.g. last ~15% of the time range held out).
- *Vehicle split*: GroupKFold by `vhcl_anon_id` (held-out vehicles), as README demands "held-out vehicles / periods".
- Final headline number: held-out vehicles AND after-cutoff period.

**Nowcast task (A1)**: simulate the real situation. For every held-out return/handover event j, pretend our last known reading is the one i reporting-lag L days earlier (L in {1, 7, 14, 28}); predict km at event j's date. Target `delta = km_j - km_i`; report error on the absolute odometer (same as error on delta).
**Forecast task (A2)**: from as-of T, predict km(T+h) for h in {7, 14, 30, 60, 90}. Truth = interpolated/next observed reading near T+h (only use pairs where an observation exists within +-3 days or use the next observed reading with its date).
**Metrics**: MAE (km), MAPE (guarded for small denominators), WAPE = sum|err|/sum|true delta|, median AE, bias (mean signed error), per horizon/lag. Plus fleet-level aggregate error (sum over vehicles), which matters for defleet planning. Also **calibration of the intervals** (coverage of the 80%/90% band).
**Baselines**: (b0) carry-forward last reading; (b1) `starter.py` (avg daily rate so far x gap); (b2) fleet/segment median rate (by vehicle_level x rental_product) x gap. Report skill = 1 - MAE_model / MAE_baseline.

**Flagging task (B) evaluation** (no labels, so three lenses):
1. *Synthetic injection* on held-out vehicles: inject known anomalies (odometer rollback, dropped digit e.g. 12345->1234, extra digit, +/-X km offsets, repeated stale value on a long rental, implausible jump, ×1.6 miles/km mix-up). Report precision / recall / PR-AUC per anomaly type and overall. Clearly say these are synthetic.
2. *Rule-proxy labels*: agreement of the model flags with physical-impossibility rules (decrease, >max plausible km/day, > vhcl_max_mileage). Report overlap and what the model catches beyond rules.
3. *Manual plausibility review*: show top-N flagged real cases with a plot (history, prediction band, flagged point) and a one-line explanation. This is the demo material.

## 3. Modelling plan

### 3.1 Feature engineering (all computed "as of" the last known reading i; strictly past data)
- Gap: days since last reading (`gap_days`), horizon h, day-of-week/month of target date (seasonality), holiday proximity (optional: public holiday library).
- Usage history of the vehicle: avg rate over last 1/3/5 rentals, last 30/90/180 days, EWM rate, overall rate since first reading, rate std, share of days on rent, number of readings.
- Rental state at as-of time: on_rent, rental_product (ShortTerm/LongTerm...), one_way flag, days since handover, planned return date -> planned days remaining (careful: check whether `rntl_return_date` is known in advance or is the actual return; if actual -> leakage, do not use for nowcast features), repair_status, status_code (defleet vehicles drive little).
- Vehicle: vehicle_level, vehicle_type, model_year, vehicle age, power_ps, cubic_capacity, driven_wheels, current km, km remaining to `vhcl_max_mileage`.
- Location/segment: branch pool, region, state (high-cardinality, salt-hashed: use as categorical in LightGBM or target-encode with out-of-fold), `uti_plan` (planned utilisation: an expected busyness signal).
- Cold start: vehicles with 1-2 readings fall back to segment-level rates (hierarchical shrinkage: vehicle rate shrunk towards segment rate by n_obs).

### 3.2 Models (cheap to expensive; each one is shippable)
| # | Model | Why |
|---|---|---|
| M0 | baselines b0-b2 | required comparison |
| M1 | Ridge / Huber on log-rate features | fast linear sanity check |
| M2 | **LightGBM** on target `rate = delta / gap_days` (objective L1/Huber), prediction = last_km + rate_hat x gap | main model; handles categoricals, missing values, nonlinearity |
| M3 | LightGBM quantile models (q10, q50, q90) or conformal residual bands (split-conformal on a calibration fold) | gives prediction intervals for the flagging and for "uncertainty on the dashboard" |
| M4 (stretch) | Direct multi-horizon: one GBM with horizon `h` as feature; vs recursive | forecast quality |
| M5 (stretch) | Mixed: M2 with per-vehicle Kalman/EWM rate state as input feature | smoother rate tracking |
Choose by held-out MAE, tie-break on simplicity. If LightGBM is not installable, use sklearn `HistGradientBoostingRegressor` (same idea, supports quantile loss).

Why predict *rate* (km/day) not absolute km: monotonic by construction when rate >= 0, scale-free across vehicles, and the error budget grows with the gap, as it should.

### 3.3 ML procedures to use (and what to skip within 7h)
- **Preprocessing**: dedupe, sort, enforce a monotone "clean odometer" for training targets (drop or repair the rule-flagged bad readings so the predictor is not trained on garbage, and so flags are not self-fulfilling), cap/winsorize rates at the 99.9th percentile, log1p on rates for stability if skewed.
- **Augmentation** (cheap, high value): *synthetic reporting lag*: re-sample (last-known, target) pairs with random lag 1-28 days from every vehicle history, which multiplies training pairs enormously and matches the production situation. Anomaly injection for evaluation (and optionally as weak supervision for a classifier).
- **Feature selection**: LightGBM gain importance + permutation importance on held-out vehicles. Ablation table (history features / rental state / vehicle attributes / location) is a great "results" slide.
- **Hyperparameter tuning**: small Optuna or random search (30-50 trials, GroupKFold by vehicle, early stopping). Cap at ~20 min total; defaults are usually within a few % of tuned.
- **Calibration**: check interval coverage on held-out data; adjust with conformal correction so a "90% band" really covers ~90%.
- **Skip**: deep sequence models (LSTM/Transformer/TFT), heavy hyperparameter searches, any LLM in the loop for the core predictions. An LLM can be used for *explanations of flags* ("why is this reading suspicious") only if time remains, it is not needed for the score.

### 3.4 Erroneous mileage detection (B)
Combine two layers into one `anomaly_score` + a reason string:
- **Rule layer (high precision, explainable)**: (R1) mileage decreases vs previous reading; (R2) implied km/day above physical cap (e.g. > 1000, tune from data quantiles / rental length); (R3) mileage > `vhcl_max_mileage` or sudden reset; (R4) return km != next handover km beyond tolerance (inconsistency between consecutive records); (R5) zero/near-zero movement over a long rental (stale odometer); (R6) round-number or repeated values (optional).
- **Model layer (catches subtle cases)**: residual `r = actual - predicted`, standardised by the quantile band or a robust scale (MAD) per segment and gap; flag |z| > threshold chosen from the injection experiment to reach a target precision (e.g. P >= 0.8) with maximum recall. Optional stretch: IsolationForest on [rate, z, gap, vehicle features] as a second, label-free signal.
- Threshold selection and the precision/recall trade-off curve go into the results. Be honest: injected anomalies are an upper bound on real-world performance.
- Important design point: for "predict then flag" use a prediction made **without** the suspect report (out-of-fold / as-of prior), otherwise the model sees the anomaly.

## 4. Repo layout (target)
```
Project_2/
  CLAUDE.md  PLAN.md  SUBMISSION.md  requirements.txt  setup.sh  starter.py (untouched)
  src/
    data.py        load, clean, observation table
    features.py    as-of feature builder (lag/horizon sampling)
    baselines.py   b0, b1, b2
    models.py      M1-M3 train / predict / intervals
    evaluate.py    splits, metrics, per-horizon tables, calibration
    anomalies.py   rules, residual flagging, synthetic injection, PR curves
    run_all.py     reproduces everything -> results/
  results/         metrics.csv, plots/*.png, flagged_examples.csv
  submission/      slides.pdf (or demo.mp4)
  data/            (git-ignored) full CSV
```

## 5. Timeline (hacking 09:30-13:00, 13:45-16:45)
| Time | Goal | Output |
|---|---|---|
| 09:30-10:15 | Load full data, EDA, verify row semantics, build observation table | `data.py`, EDA notes in PLAN/SUBMISSION |
| 10:15-10:45 | Freeze eval protocol: splits, lag sampler, metrics; run baselines b0-b2 | first results table (the bar to beat) |
| 10:45-12:15 | Feature builder + LightGBM M2 on nowcast; first beat-the-baseline | v1 numbers, commit |
| 12:15-13:00 | Rule layer (B) + synthetic injection harness; start forecast horizons | rules + first PR table |
| 12:45-13:00 | **Freeze features + eval protocol**, commit, launch HPO script in background (wall-clock cap ~35 min, resumable, GroupKFold by vehicle, MAE objective, trials logged to `results/hpo/`) | `results/hpo/best_params.json` by 13:45 |
| 13:45-14:45 | Review HPO result (tuned vs default), quantile/conformal bands, residual flagging, forecast horizons, ablations | main results complete |
| 14:45-15:30 | Error analysis (worst vehicles, long gaps, cold start), quick tuning, fix leaks, plots | final numbers |
| 15:30-16:15 | `SUBMISSION.md`, 5 slides (or demo video), clean `run_all.py`, fresh-clone run test | submission ready |
| 16:15-16:45 | **Code freeze**, push, check repo is public, rehearse 3-min pitch | done |

If 2-3 people: person 1 = data + A (nowcast/forecast), person 2 = B (rules, injection, residual flags), person 3 (optional) = evaluation harness, plots, slides/pitch. Agree on the interface early: `observations.parquet` + a function `predict(vehicle_df, as_of, horizon) -> (median, lo, hi)`.

## 6. Risks and fallbacks
- *Data semantics unclear* -> ask the owners early (they are on site / reachable); fall back to using only `NotOnRent` rows as truth readings.
- *Full data large / slow* -> subsample vehicles for dev (e.g. 10%), run full only for final numbers.
- *Model barely beats baseline* -> still a valid honest result; improve on the segments where it helps (long gaps, long-term rentals) and show where the baseline fails.
- *Leakage* (`rntl_return_date` actual vs planned, future status codes) -> test by dropping suspicious features and checking the score doesn't collapse.
- *Time overrun* -> M2 + rules + injection eval is the minimum viable submission; M3 bands, multi-horizon and ablations are stretch.

## 7. Business framing (for the pitch / "Business relevance" 15 pts)
- Nowcast lets valuation and defleet planning use today's km instead of a month-old report, quantified as MAE reduction vs carry-forward and vs starter baseline.
- Forecast km at planned defleet date -> choose sale timing before contract/max-mileage limits, flag cars about to exceed `vhcl_max_mileage` (excess-km fees).
- Flags route to a human review queue with a reason, so wrong or manipulated reports are caught early. Next steps: calibrate on real confirmed errors, integrate as a nightly batch job, add telematics where available.

---

# PART II (v2, 13 Nov, after the first real-data results): model, design decision, optimisation roadmap

## 8. What is actually built (status quo, honest)

**Model.** One LightGBM regressor (`src/run_forecast.py`), objective L1, target = **km/day rate** `y_rate = (km_t - km0) / gap`, sample weight = `gap`. Prediction = `km0 + rate_hat * gap`.
Weighting an L1 loss on the rate by `gap` makes the loss exactly `|km_hat - km_t|`: we minimise MAE in km directly, while the target stays scale-free across horizons.

**Samples (direct multi-horizon, no recursion).** For many pseudo-cutoffs C (every 30 days from 2023-03-01) and every training vehicle: freeze everything known at C, sample up to 8 later readings t (<= 150 days ahead, never beyond the validation cutoff) as targets. `gap = t - last_day` (includes the staleness of the last report), `h = t - C`. This mimics the test exactly: last known reading can be weeks old, targets 1-147 days later, no rental status at t.

**Features (as of C only).** Last km, staleness, rates over 7/30/90/180/365 d, cumulative rate, #readings in 90 d, history length, km left to max mileage, rental state at the last reading (on rent, product, one-way, days since handover, days to return date), status code, uti_plan, static vehicle attributes, branch pool/region/state, target month/day-of-year, a fleet seasonal factor fitted on the training period.

**Cleaning.** One reading per vehicle-day (max). Two-sided outlier rule (a reading breaks monotonicity vs the median of its 3 previous / 3 next readings by > 100 km, or absurd values). Clean series forced monotone with `cummax`. Training targets: clean readings only. Validation targets: reported with and without flagged readings.

**Validation.** Held-out 20% of vehicles, cutoff 2025-07-01 (same season as the real test cutoff 2026-07-01), targets until 2025-11-11 (same span as the test). 34.6k target readings.

| model (held-out vehicles, all targets) | MAE km | MedAE | WAPE | bias |
|---|---|---|---|---|
| b0 carry-forward | 5,486 | 4,989 | 1.00 | -5,414 |
| b1 vehicle cumulative rate (≈ starter.py) | 2,200 | 1,324 | 0.40 | -320 |
| b1b vehicle 90-day rate | 2,335 | 1,405 | 0.43 | -56 |
| b2 segment pooled rate | 1,680 | 1,158 | 0.31 | -547 |
| **LightGBM** | **1,438** | **949** | **0.26** | -382 |
| LightGBM without location features | 1,487 | 1,023 | 0.27 | -230 |

-35% MAE vs the starter-like baseline, -14% vs the best naive baseline. The gain grows with the gap (91-150 d: 1,999 vs 2,357 segment vs 3,395 starter).

**Three facts in these numbers that drive the next steps:**
1. **The segment rate beats the vehicle's own rate** (1,680 vs 2,200). A rental car's past usage is a weak predictor of its future usage (different customers each time). The signal sits mostly at *fleet/segment/location level*; the vehicle's own history only adds a correction. => shrinkage and group-level rate features are the most promising feature work.
2. **Systematic negative bias** (-382 km overall, -628 km at 91-150 d). Partly expected (L1 predicts the median, the rate distribution is right-skewed), partly a level shift (Jul-Nov 2025 drove more than the training average). => cheap bias calibration + "recent fleet level" features.
3. **Short gaps have a high relative error** (WAPE 0.54 at 1-14 d): what the car does *right now* (on rent or parked, long- or short-term) dominates, history averages say little. => rental-state features matter most for the nowcast.

Checked on the test file: in the last known row per test vehicle, 83% are OnRent, 15% LongTerm, 5% status 3 (defleet), and 25% carry a return date after 2026-07-01. ~~So `ret_in` is not leakage w.r.t. the test task.~~ Superseded (section 14): the return date is the *actual* one, so for open rentals it is future information; `ret_in` is dropped from the headline model.

## 9. One model or two? (decision)

**Decision: one prediction engine used in two modes, plus a thin rule layer. No separate anomaly ML model.**

```
              +-----------------------------------------------+
 history ---> | ENGINE: expected km at date t given history   |---> A. nowcast / forecast (median + 80/90% band)
 up to as-of  | LightGBM rate (q50) + band (quantile/conformal)|
              +-----------------------------------------------+
                             | same engine, as-of = day BEFORE the report
                             v
 report r --> residual z = (km_r - pred) / band width ----+
          --> two-sided consistency (prev AND next reads) +--> anomaly score + reason + tier  (B)
          --> physical rules (decrease, > cap km/d, > max) +
```

Why not two separate models:
- **There are no labels for B.** A separately trained anomaly classifier could only learn from our own synthetic injections, i.e. it would learn our injection generator, not real errors. Unsupervised alternatives (IsolationForest on raw features) don't know what "expected usage" is; the forecast model does.
- **B's quality is A's quality.** A flag means "far from expected"; the better the expectation and its uncertainty, the better the flag. Every improvement to A improves B for free. One feature pipeline, one model to tune, one story for the pitch (the README itself suggests this link).
- What B needs on top is *not* a second model but **(a) an uncertainty band** (a 2,000 km deviation is huge for a parked car and normal for a long-term rental after 90 days) and **(b) the two-sided view**: when judging a historical report we also see the readings *after* it, which is far more precise than any forecast. A spike followed by readings consistent with the ones before it = typo. A jump after which all later readings stay shifted = level shift (cluster swap, or the earlier history was wrong). `series.flag_outliers` is already a primitive two-sided check.

When B *would* justify its own model: once SIXT confirms some flags (review queue => labels), train a small supervised classifier on [residual z, two-sided deviation, rule hits, gap, rental product, ...]. That is a "next step" slide, not a today task.

Caveat to state honestly: the outlier filter is used both to clean training targets and inside B. Fine for training, but in the B evaluation report rules-only, residual-only and combined separately so we don't grade our rules with our rules.

## 10. Optimisation roadmap for the engine (ordered by expected gain per hour)

Each step: `python -m src.run_forecast --frac 1.0 --tag _<name>`; keep only if held-out MAE improves; log each row in an ablation table (that table is itself a results slide).

**P1. Free gains (<= 30 min)**
- All training vehicles (`--frac 1.0`), denser cutoffs (`--step 14`), more targets per vehicle (`--max_targets 16`).
- **Bias calibration**: on inner-val, fit a multiplicative factor per gap bucket `k = median(true delta) / median(pred delta)` and apply it. Alternative: weight training cutoffs by recency (`w *= 0.5 ** age_years`) so recent behaviour dominates.
- Clip `rate_hat` to [0, p99.9 of training rates].
- Seed-average 3-5 LightGBMs (~1-2%).

**P2. Feature engineering (the real lever, 1-2 h)**, all strictly as of C:
- *Shrunk vehicle rate*: `(n_eff * rate_v + k * rate_seg) / (n_eff + k)`, n_eff = history days / 30, k ≈ 3-6 (tune). Encodes fact 1 directly; trees approximate this blend poorly.
- *Group-level recent rates (time-safe target encodings)*: pooled km/day over the last 30/90 days for the vehicle's **pool, region, vehicle_level, rental_product**, and the whole fleet, from other vehicles' readings <= C. Captures local demand and the current fleet level (fixes fact 2). Also the same group's rate *in the target month last year* (regional seasonality: Florida winter vs Northeast summer).
- *Rental-behaviour history*: share of days on rent in the last 90/365 d, number of rentals, mean rental length, share LongTerm, km/day while on rent vs while idle. Physical model: expected km = P(on rent) x on-rent rate; give the GBM those ingredients.
- *Rate robustness*: median of per-interval rates, EWM rates (half-life 14/60 d), rate CV, last-interval rate, share of zero-movement intervals.
- *Lifecycle*: age in months, km / age, `km0 / vhcl_max_mileage`, days since status became 3 (defleet cars stop driving). Cars near max mileage get defleeted => rate collapses.
- *Calendar over the forecast window*: number of US public holidays and weekend days in (last_day, t], summer share of the window.
- *State x window*: `min(ret_in, gap)` = days of the window still on the current rental.
Ablation by feature group (history / rental state / vehicle / location / calendar) => slide.

**P3. Target / loss variants (30 min)**
- Predict the rate *relative to the shrunk/segment rate* (`y_rate / base_rate` or `y_rate - base_rate`): an easier job for the GBM, helps vehicles with short histories.
- L1 vs Huber vs L2 (+bias correction), keep weight = gap, pick by held-out MAE.

**P4. Hyper-parameter search (background, <= 35 min wall clock)**
Optuna/random search: num_leaves 31-255, min_child_samples 50-1000, learning_rate 0.02-0.1, feature_fraction, lambda_l2, cat_smooth for the high-cardinality pool codes. Inner-val (vehicle-disjoint), early stopping. Expect 2-5%.

**P5. Robustness of the evaluation (must-have for the "results" points)**
- Backtest at 3 cutoffs (2024-07-01, 2025-01-01, 2025-07-01): mean ± spread of MAE, model vs baselines. Shows the gain is not one lucky season.
- Per segment: ShortTerm vs LongTerm at the cutoff, on-rent vs not, status 3, short history (< 5 readings). LongTerm with sparse reporting is the README's motivating case: show it explicitly.
- Interval coverage of the 80/90% bands per gap bucket.

**P6. Uncertainty bands (needed for B and for the business story)**
LightGBM quantile heads (alpha 0.1/0.9) on the same samples, then split-conformal correction per gap bucket on inner-val so coverage is exact. Output `(km_lo, km_med, km_hi)`.

**P7. Test-set predictions (the actual deliverable of A)**
Retrain on *all* labelled vehicles (test vehicles are disjoint), cutoff 2026-07-01, predict every masked test row -> `results/test_predictions.csv` (vehicle, date, km_pred, lo, hi). **Cold start**: ~3.7k masked rows belong to vehicles with no known reading, `build_samples` drops them today. Fallback: segment-level km-by-age (model year, vehicle_level, vehicle_type) or a no-history model on static + calendar features; mark them low-confidence.

Skip today (and say so deliberately in the pitch): deep sequence models (TFT, N-BEATS, LSTM), per-vehicle ARIMA. With ~100 noisy irregular readings per vehicle and the signal sitting at fleet level, a GBM on good aggregate features is the right tool.

## 11. Anomaly detection (B), concrete implementation

`src/anomalies.py` (to write):
1. **Rules** (each with a reason string and severity): R1 decrease vs previous clean reading > 100 km; R2 implied km/day > cap (p99.9 of clean interval rates per rental product); R3 km > 1.1 x vhcl_max_mileage or >= 900k; R4 zero / near-zero after a non-zero history (reset); R5 stuck: identical km over a long on-rent period (> 30 d); R6 same-day conflicting km beyond tolerance.
2. **Residual score**: engine prediction from history strictly before r (as-of = previous reading), `z = (km_r - q50) / ((q90 - q10) / 2.56)`. Two-sided score: deviation from the interpolation between the clean previous and next readings, scaled by the gap. Score = max(rule severity, |z_forward|, |z_two_sided|); the dominant term becomes the reason.
3. **Tiers**: *certain* (rule hit), *likely* (|z| above the threshold giving precision >= 0.8 on injections), *watch* (moderate z, or forward-only evidence because there is no next reading yet = the real-time case).
4. **Evaluation by synthetic injection** on held-out vehicles (1-2% of reports): rollback, dropped digit (12345 -> 1234), extra digit / x10, swapped adjacent digits, +/- 2-20k km offset, stuck odometer (repeat previous value), miles<->km mix-up (x1.609, /1.609; plausible for a US fleet). Precision / recall / PR-AUC per type for rules-only, residual-only, combined. State clearly: injected = upper bound for real performance.
5. **Real-data plausibility**: top-20 real flags with a small plot each (history, band, flagged point, reason). This is the "how plausible are the flags" criterion; pick examples that explain themselves.

## 12. "Creative" next steps (pitch material: each is cheap once the engine exists)

Ranked by business value x feasibility. #1-#3 are doable today in < 45 min each; the rest are roadmap slides.

1. **Typo auto-correction, not just a flag.** For a flagged report, test whether a simple edit explains it: drop/insert a digit, swap adjacent digits, x1.609 / /1.609. If the edited value lands inside the expected band: "probably a typo, suggested value 41,532 (band 40,900-42,300)". Turns a review queue into a one-click fix. Measurable on injected typos (correction accuracy).
2. **Max-mileage crossing date.** Engine + band => *when* each car hits `vhcl_max_mileage` (date + interval). "These N cars will exceed their max mileage before their planned defleet" is a direct defleet-planning tool. Validate by backtest on cars that crossed after the cutoff.
3. **Value-of-information reporting policy.** The band width says *which* long-term cars we are uncertain about. Ask for an odometer photo/reading only when the band (or the implied valuation error in €) is wide, instead of chasing every monthly-reporting customer. Simulate: "revealing the true reading for the 10% most uncertain cars removes X% of total fleet error".
4. **Fleet-level aggregation.** Sum predictions per pool/region: errors partly cancel, so fleet-level km nowcast error is far smaller than per-car error; relevant for finance/valuation.
5. **Rental-level (customer) view.** Long-term rentals whose reported km/day sits far below the segment and the vehicle's other rentals (possible under-reporting before return).
6. **Human-in-the-loop labels.** Reviewer decisions (confirmed error / fine) become labels -> supervised flag model, re-tuned thresholds; the system improves monthly.
7. **Telematics / OEM odometer feeds** as ground truth where available; the model covers the rest of the fleet.

## 13. Remaining time-box for today
1. P1 + bias calibration + 3-cutoff backtest -> new headline table.
2. P2 group-level and shrunk rates (biggest expected gain) -> ablation table.
3. P6 bands + `anomalies.py` (rules, residual z, injection PR table) -> B results.
4. P7 test predictions + cold-start fallback.
5. Creative #1 (typo correction) and #2 (max-mileage date) if time remains; #3 as a slide.
6. SUBMISSION.md, slides, `run_all.py`, fresh-clone run.

## 14. Diagnosis of the -400 km bias and the "features don't help" plateau (13 Nov, afternoon)

Observed: new features (group rates, shrunk rate, rental shares) -> MAE 1,444 (vs 1,438), bias -430. Per-gap calibration factors on the inner slice ~1.00-1.03, yet the held-out window is ~400 km low. Fleet km/day in Jul-Nov 2025 = 2024. Short intervals between readings show much higher km/day than long ones.

**Root cause: the number of reports a car generates is itself a strong signal of how much it drives (informative sampling), and our training sample weights it differently from the evaluation.**

Evidence (held-out window 2025-07-01..11-11, 8,166 vehicles, km/day from last reading before C to last reading in the window):

| report days in window (quintile) | vehicles | median km/day |
|---|---|---|
| 1-10 | 1,740 | 29 |
| 11-19 | 1,715 | 62 |
| 20-25 | 1,685 | 74 |
| 26-30 | 1,397 | 80 |
| 31-49 | 1,629 | 83 |

Spearman(#reports, km/day) = 0.52. Reports are triggered by rental events (handover, return, status change), so a busy car both drives more and reports more.

- **Training** samples at most 8 random targets per vehicle (`max_targets=8`) => every vehicle has ~equal weight. The **inner slice** is built the same way => no bias there.
- **Validation** (and the **test**!) score *every* reading => busy cars count ~3x more than quiet ones. Row-weighted median km/day = 76.7 vs vehicle-weighted 70.5: +6 km/day x ~65-day mean gap ≈ **400 km. This is the bias**, almost exactly.
- The "short intervals = high km/day" effect is the same phenomenon seen from the other side, not a separate artifact.
- Why the new features didn't help: they all describe the *past*. The remaining error is mostly *future* activity (will this car be rented a lot in the next 4 months?), which past usage barely predicts (fact 1 in section 8). Better past features can't fix a weighting mismatch or supply future information.

The test file has the same structure: masked target dates per vehicle mean 20.1, median 21 (validation window: 20.7 / 22). So the test is scored the same row-weighted way as our validation, and the **schedule of target dates is visible in the test file**.

### Fix 1 (clean, no debate): make training weights match the evaluation
- Use all targets per vehicle in training (`--max_targets 0`) or keep the cap and weight each sample by `n_targets_vehicle / n_sampled` (x gap as now). Same for the inner slice so early stopping/calibration see the right distribution.
- Expected: bias ~0, MAE down a bit (median shifts to the row-weighted one). Running now: `python -m src.run_forecast --frac 1.0 --max_targets 0 --tag _nocap`.
- Same fix needed in `train_final.py`, `hpo.py`, `coldstart.py` (all use `max_targets=8`).

### Fix 2 (big lever, needs a decision): "report-schedule" features
The dates at which the test asks for predictions are known inputs. From them, per target row:
- `n_tgt_window`: number of target dates of this vehicle in (C, C+150]; `tgt_per_30d` = density.
- `n_tgt_before_t`: number of target dates in (last_day, t] (= how many rental events happened before t), and per day of gap.
- `d_prev_tgt`, `d_next_tgt`: days to the neighbouring target dates (dense = active rental churn, a long hole = parked or long-term rental).
- `last_tgt_day - C`: when the last report appears (a car whose reports stop early may be defleeted/sold).
In training, compute exactly the same from the real readings after each pseudo-cutoff (their dates only, never their km). Given the table above (29 vs 83 km/day), this should beat every history feature we have.

**Is it legitimate?**
- *For this test file:* yes, the dates are part of the input; nothing about the km is used.
- *For production:* the exact future report dates are not known, but the closest real equivalent is: **SIXT knows its future bookings** (reservation system: handovers/returns scheduled in the window, long-term contract end dates). Pitch it as "event-aware forecasting: booked rental days in the window"; the report schedule is our proxy for the booking calendar.
- *Honest reporting:* show both numbers side by side: **history-only model** (strict nowcast, valid at any time) and **schedule-aware model** (uses known future event dates). Never mix them in one headline.

### Other consequences
- Report MAE both row-weighted (matches the test) and vehicle-weighted (each car counts once, the fleet-valuation view). The baselines must use the same weighting.
- For B (anomalies): the residual z of a report should use the schedule-aware prediction when judging historical reports (the report dates around it are known then anyway).
- Business story: "a car's rental activity, not its past mileage, drives its future mileage" is a strong insight slide, with the quintile table above.

### Decisions (13 Nov, team)
- **Fix 2 is rejected.** We do not derive any feature from the masked test rows (not even their dates): the test file's future part is the test, not an input. Only the known history up to the cutoff is used. The informative-sampling finding stays as an *insight* (slide + explanation of the bias), and Fix 1 handles the weighting.
- **`rntl_return_date` is treated as the ACTUAL return date** (nearly all dates lie before today, 13 Nov 2026). For a rental still open at the cutoff, its return date is future information => `ret_in` and `ret_in_gap` are leakage for a real nowcast and must be dropped from the headline model. Handover date (`since_handover`) stays: it is in the past. Ablation `--drop ret_in,ret_in_gap` is running to measure how much the model leaned on it; expect the MAE to get somewhat worse, which is the honest number.
- Production substitute (roadmap, not today): the *planned* return date / contract end from the booking system, which SIXT does know at the cutoff.

## 15. HPO and optimisation after the model-family comparison (13 Nov, 12:15)

Ridge 1,492 / HistGB 1,434 / LightGBM 1,426 km on the same split: model capacity is not the bottleneck. Expect HPO to give 1-3%, not more; it is worth running over the break but does not drive the story.

**HPO review (fixed in `src/hpo.py`):**
1. *Wrong search slice*: it used inner cutoffs in the last 150 days before the val cutoff; their targets are capped at the cutoff, so the search optimised short gaps (1-60 d) while the MAE is dominated by 60-150 d gaps. Now: cutoffs 150-365 d before the val cutoff (full horizon, includes the same season one year earlier).
2. *Row-count params not transferable*: tuned on a 600k-row subsample, applied to ~2x+ more rows. `min_child_samples` / `min_data_per_group` are now rescaled (`final_params`).
3. *Missing the parameters that matter here*: the pool code (high-cardinality categorical) carries ~40% of the gain; added `max_cat_threshold`, `min_data_per_group`, wider `cat_smooth`, and `min_split_gain`; num_leaves up to 511.
4. *Resume re-queued the default trial* each time: now only on a new study. Multivariate TPE. Per-trial progress log, warning spam silenced.
5. *Held-out check not comparable to run_forecast*: predictions now clipped to [0, p99.9 rate] like run_forecast; bias reported.
6. `train_final.py --params <json>` applies the tuned params (off by default, the shipped predictions are unchanged).
Smoke test (10% of vehicles, 50k rows, 1 min): 39 trials, inner MAE 1,490 vs default 1,521 (-2%), consistent with the expectation.

**Optimisation methods ranked for the remaining time (beyond HPO):**
1. *Ensemble of the three families* (mean or inner-fitted weights of LightGBM + HistGB + Ridge): their errors differ (Ridge bias +245, GBMs -120), so averaging typically gives 1-3% for free. Cheapest real gain.
2. *Seed / bagging average* of 5 LightGBMs (different seeds + vehicle bootstraps): ~1%, also stabilises the result.
3. *Per-vehicle random effect via target encoding of the vehicle's own past residuals*: the model's residual on the vehicle's earlier cutoffs (time-safe) as a feature. Captures "this car always runs hotter than its pool" better than raw rates.
4. *Quantile heads + conformal bands*: no MAE gain, but required for anomaly flagging and the business story (uncertainty per car).
5. *Multi-cutoff headline*: report mean ± spread over 3 cutoffs (results credibility > a 1% gain).
**Result (12:15, `src.compare_models`, 50% of training vehicles, `logs/compare_ens.txt`)**: LightGBM 1,426; 4-seed bag 1,423; mean of the two GBMs 1,424; mean of all 3 1,428 (bias -3 but per-vehicle MAE worse); convex blend with inner-fitted weights (ridge 0.10 / HistGB 0.10 / bag 0.80) 1,422. => ensembling gains 0.3%, within noise. **Decision: ship the single LightGBM** (simpler, reproducible); mention the ensemble test as "tried, saturated". This, plus the Ridge result, is strong evidence that the model side is done and the limit is information, not capacity.

Do not spend more time on: ensembles/bagging (tested, +0.3%), more history features, deeper models, denser cutoffs (all tested, no gain). The remaining error is future rental activity, which is not in any input; say so explicitly in SUBMISSION.md, together with the insight table from section 14.

### Open questions (for us / for the owners)
1. May the target-date schedule be used as a feature for the test predictions? (Ask Julian/Ernest; if unclear, ship both models and label them.)
2. Is `rntl_return_date` the planned or the actual return date? If actual, it is future information in production (fine for the test file, but say so).
3. Will the test be scored on all masked rows (row-weighted, as we assume) or per vehicle?
