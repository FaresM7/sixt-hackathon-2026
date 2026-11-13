# CLAUDE.md: SIXT Hackathon 2026, Project 2 (Mileage Forecasting & Erroneous Mileage Detection)

## Context
- One-day hackathon, Fri 13 Nov 2026. Hacking 09:30-13:00 and 13:45-16:45. **Submission freeze 16:45** (we self-freeze code at 16:15).
- Work ONLY in `Project_2/`. Public/mock data only. No secrets in git (`.env` is git-ignored). No branches in the shared repo; work on our own fork/clone.
- Judged (100 pts): problem understanding 15, approach quality 25 (vs. baseline!), results 25 (measured, honest), business relevance 15, code/reproducibility 10, presentation/demo 10.
- Goal is *a working, honestly evaluated* solution, not a perfect one. Always keep an end-to-end runnable pipeline; improve incrementally.

## Task
- **A. Nowcast/forecast** odometer (`vhcl_current_mileage`) of a vehicle at "now" (as-of date T, last report up to 28 days old) and at T+h.
- **B. Erroneous mileage detection**: flag reports inconsistent with expected usage. No labels exist: residual-based flagging + rule-based proxy labels + synthetic anomaly injection for evaluation.
- Metrics: MAE / MAPE (also WAPE, bias) on held-out vehicles AND held-out time periods, vs naive baseline (`starter.py`); plausibility of flags.

## Data (full CSV arrives on the day; put in `data/`, git-ignored)
One row per status update. Columns: see README.md. Key facts learned from `example_file.csv` (1 vehicle, 57 rows):
- File is NOT sorted; several rows per day; the same date can show different mileages. Always sort by (vehicle, date, mileage).
- **Row semantics are only partly "OnRent = handover km"**: true for the example vehicle, but on the real data only 79% of rentals with several rows have constant km (the rest vary by hundreds to thousands of km). Do not rely on it; ask Julian/Ernest.

## Real data (in `data/`, parquet; read with `pd.read_parquet`, needs `pyarrow`)
- `data_30_percent_sample.parquet`: 2.70M rows, 27,630 vehicles, 2022-01-02 .. 2026-07-01, fully labelled. ~98 rows/vehicle.
- `test_data.parquet`: 656,601 rows, 8,254 DIFFERENT vehicles (no overlap with train). Known rows end 2026-07-01; ~25% of rows are **masked** (mileage/status/rental fields NaN, only vehicle static columns + date), dated 2026-07-02 .. 2026-11-11, ~20 per vehicle, always at the tail of each vehicle's timeline. => the real task is a **forecast of km at given future dates, 1-147 days (median 67) after the last known reading**, with no rental status known at the target date. ~3.7k masked rows belong to vehicles with no known row (cold start).
- `sixt_predicted_mileage_interp` and `sixt_error` are 100% NaN in both files (meaning unknown, ask owners; maybe SIXT's own estimate/error to compare against).
- No labels for the test rows: validate with pseudo-cutoffs inside the train file (hide everything after a cutoff C for held-out vehicles, predict the real later rows).
- Noise: 0.7% of daily-max readings go DOWN (median drop 1.5k km, 1% quantile -84k km), 173 zero readings, a 999999 value, 0.24% above `vhcl_max_mileage`, up to 5 rows per vehicle-day with different km. Median 50 km/day (daily series), per-vehicle average 70 km/day (IQR 59-82). Gap between consecutive reading dates: median 3 days, p99 30 days. LongTerm = 4.4% of rows.
- `NotOnRent` rows have null handover/return dates. `vhcl_max_mileage` can change over time (25000 -> 24000). `status_code` 3 (defleet) has `uti_plan` = 0. First reading of a new car is ~5 km.
- Example vehicle: ~70 km/day average, only ShortTerm rentals. The full data should contain long-term rentals with sparse reporting.

## Implemented so far
- `src/series.py`: `prepare(parquet)` -> (all daily readings with `ok` outlier flag, clean monotone series). One reading per vehicle-day (max km); outlier = breaks monotonicity vs the median of 3 neighbours by >100 km, or absurd (>=900k, >3x max mileage).
- `src/features.py`: `build_samples` makes (vehicle, cutoff, target day) rows with as-of features (last km, staleness, rates over 7/30/90/180/365 d, cumulative rate, rental state, static attrs, location, seasonal factor); target = km rate per day.
- `src/baselines.py`: b0 carry-forward, b1 cumulative rate, b1b 90-day rate, b2 segment pooled rate.
- `src/run_forecast.py`: validation at a pseudo-cutoff (default 2025-07-01, held-out 20% vehicles, targets to 2025-11-11) + LightGBM (L1 on rate, weight=gap). Run from `Project_2/`: `..\.venv\Scripts\python.exe -m src.run_forecast --frac 1.0 --tag _all`; flags `--drop f1,f2` (ablation), `--step`, `--cutoff`.
- Results so far (held-out vehicles, 2025-07-01 cutoff): LightGBM MAE ~1,438 km vs best naive (segment rate) 1,680 km; no-location-features ablation 1,487 km. See `results/forecast_val_*.csv`.
- `src/train_final.py`: final fit on ALL labelled vehicles (cutoff 2026-07-01; #trees chosen on a 10% vehicle slice, then refit on everything) + predictions for all 163,047 masked test rows -> `results/test_predictions.csv` (git-ignored; no true values exist, so it is not scored). Includes `src/coldstart.py` model for the 149 vehicles (3,745 rows) without history (NOT validated yet). Run: `python -m src.train_final` (~12 min).
- **Robustness / segments / bands / cold start (14:45)**: tuned MAE at 3 cutoffs 1,560 / 1,379 / 1,416 (mean 1,452) vs segment 1,837 / 1,564 / 1,680 (-12..-16%). Segments: LongTerm 2,052 vs 2,278; last reading > 28 d old 1,722 vs 3,291. 80% bands (quantile LGBM + conformal per gap bucket, `--bands`): coverage 0.82. Cold start (`src/eval_coldstart.py`): 2,465 vs 2,763. Slide charts: `python -m src.make_charts`. SUBMISSION.md is complete except slides.
- **Task B done (14:05), `src/anomalies.py`** (`python -m src.anomalies`, ~2 min; `--no_forward` skips the model part): rules (absurd, reset, drop, rate > p99.9 = 3,018 km/d; drop/rate only on the culprit reading) + two-sided score (5 neighbours per side, two passes, not blamed if consistent with one side: AGREE_KM 300 + 150 km/day) + forward score with the task-A LightGBM (real-time mode). Injection eval on held-out vehicles (1%, 7 error types): rules+two-sided flags 0.5% of readings, recall 0.92 for errors >= 1,000 km, precision lower bound 0.57; starter.py rules: 2.6% flagged, 0.93, 0.21. Stuck odometer 0% (undetectable). Typo auto-correction exact: miles 98%, swap 97%, extra digit 54%, dropped digit 37%. Real data: 0.49% flagged. Outputs: `results/anomaly_eval.csv`, `anomaly_correction.csv`, `anomaly_flags_top.csv`, `plots/anomaly_examples.png`.
- **HPO done (13:28)**: 91 trials / 60 min; held-out MAE default 1,427 -> tuned 1,416 km (-0.8%), bias -196. Params in `results/hpo/hpo_best.json` (`final_params`); test predictions regenerated with `python -m src.train_final --params results/hpo/hpo_best.json` (log `logs/final_tuned.txt`).
- Ensembles (`src/compare_models.py --bags 3`, 12:15): 4-seed bag 1,423, blend with inner-fitted weights 1,422 vs single LightGBM 1,426 => +0.3%, not worth shipping; single LightGBM stays. Dropping the return-date features cost nothing (1,429 vs 1,431). `results/README.md` indexes all result files; superseded runs are in `results/archive/`.
- `src/hpo.py` (reviewed + fixed 12:15, smoke-tested): Optuna TPE (multivariate), time-capped, resumable via `results/hpo/<tag>.db`. Search slice = inner vehicles at cutoffs 150-365 d before the val cutoff (full horizon; the old slice used the last 150 d of cutoffs, whose targets are truncated to short gaps). Also tunes categorical regularisation (cat_smooth, cat_l2, max_cat_threshold, min_data_per_group: pool code carries ~40% of the gain). Row-count params are rescaled from the trial subsample to the full data (`final_params` in `<tag>_best.json`). Held-out check clips like run_forecast, reports bias. Run: `python -m src.hpo --minutes 30 --tag hpo` (~10 min setup + 30 min search + ~10 min held-out fits); apply with `python -m src.train_final --params results/hpo/hpo_best.json`. Denser cutoffs (step 14, 16 targets) gave no gain (MAE 1,440 vs 1,438), keep step 30 / 8 targets.
- **Current headline (12:10, weighted training, no ret_in, base features)**: held-out vehicles, cutoff 2025-07-01: LightGBM MAE 1,427 km (bias -194, per-vehicle MAE 1,418) vs best naive (segment rate) 1,680. `--with_new` group-rate features gave no gain at 3 cutoffs (off by default). Model families on the same split (`src/compare_models.py`, 50% of vehicles): Ridge 1,492 / HistGB 1,434 / LightGBM 1,426 -> the linear model is only ~5% behind; most of the signal is simple. `results/test_predictions.csv` regenerated with this setup.
- Note: WAPE of b0 is 1.0 by construction; baselines are our own naive ones, not SIXT's.

## Design decisions (details in PLAN.md Part II, sections 8-13)
- **One prediction engine for A and B**, no separate anomaly ML model (no labels). B = physical rules + residual z vs the engine's prediction made from history strictly before the report + two-sided consistency (previous AND next readings) -> score, reason, tier.
- Segment rate beats the vehicle's own rate => next feature work = shrunk vehicle rate + time-safe group-level recent rates (pool/region/level/product/fleet). Model has a negative bias (-382 km, -628 at 91-150 d) => bias calibration on inner-val.
- **Bias root cause (PLAN.md section 14)**: #reports per car correlates with its km/day (Spearman 0.52; 29 km/d for cars with <=10 reports in the Jul-Nov window vs 83 for 31+). Training caps 8 targets per vehicle (equal vehicle weight); validation and test score every reading (busy cars weigh ~3x) => ~-400 km bias. Fix 1: train with all targets / weight by target count (`--max_targets 0`). Fix 2 (features from the masked test dates) is REJECTED by the team: nothing from the masked test rows is used as input.
- **`rntl_return_date` = actual return date** => `ret_in` / `ret_in_gap` leak the future for rentals open at the cutoff; drop them from the headline model (`--drop ret_in,ret_in_gap`). `since_handover` is fine.
- Next: bands (quantile + conformal), `src/anomalies.py` with synthetic injection eval, test predictions incl. cold start, creative extras (typo auto-correction, max-mileage crossing date, value-of-information reporting policy).

## Conventions
- Python 3.11+, deps in `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, pyarrow, lightgbm), installed in the root `.venv`.
- **No leakage**: features for a prediction at as-of date T may only use information available at T. Splits are by time (train < cutoff <= test) and by vehicle (held-out vehicles). Never random row splits.
- Fixed random seeds. One command reproduces all numbers: `python -m src.run_all` (target layout in PLAN.md).
- Keep notebooks out of the submission; scripts + a `results/` folder with metrics tables and plots.
- Do not edit `starter.py` (keep as the baseline reference); new code goes in `src/`.

## Deliverables checklist (by 16:45)
- [ ] code in `Project_2/`, runs from `SUBMISSION.md` (template: `../SUBMISSION_TEMPLATE.md`, must state "Project 2")
- [ ] `submission/slides.pdf` (max 5) or `submission/demo.mp4` (<50 MB, 1 min)
- [ ] evaluation results table with baseline comparison, honest limitations
- [ ] repo public, link = the one in the registration form
- [ ] 3-min pitch prepared (in case of finalist selection)

See `PLAN.md` for the approach, evaluation design and timeline.
