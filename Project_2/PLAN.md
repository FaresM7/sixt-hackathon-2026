# PLAN: Project 2, Mileage Nowcast / Forecast / Anomaly Flagging

Draft v1 (written before seeing the full data). Revise after the first 30 min of EDA.

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
