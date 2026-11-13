# Submission: Project 2, <The golden butterfly>

> DRAFT (13 Nov, 12:40). Numbers marked TODO are still being produced (HPO, anomaly detection).

## Authors
**Team name:** <The golden butterfly>

**Project:** Project 2: Mileage Forecasting & Erroneous Mileage Detection

## Problem and approach
SIXT needs a vehicle's current and future odometer reading for valuation and defleet decisions, but readings arrive irregularly (long-term rentals report up to weeks late). The test file reflects this: the last known reading is on 2026-07-01, and we must predict the km at ~20 given dates per car, 1-147 days later (median 67), with no rental status known at those dates.

**Main idea: predict a usage rate, not an odometer value.**
`km(t) = last reading + predicted km/day x days since that reading`.
A gradient-boosted model (LightGBM) learns the km/day from what is known at the cutoff: the car's recent usage, its rental state, its type and its branch/pool. The loss is L1 on the rate weighted by the gap in days, which equals the absolute error in km, so we optimise MAE directly. Predictions are clipped to [0, p99.9 rate] and made monotone per car.

**How we train it (mimicking the test):** pick many pseudo-cutoffs in the past (every 30 days since 2023-03). For each, freeze what was known at that date, and use the car's real later readings (up to 150 days ahead) as targets. One model covers all horizons; the gap is a feature.

**Data cleaning:** one reading per car-day (the max); a reading is flagged as an outlier if it breaks monotonicity against the median of its 3 previous / 3 next readings by > 100 km, or is absurd (>= 900k km, > 3x max mileage); 1% of readings are flagged. The clean series is forced monotone. Training targets use clean readings only; we report validation both with and without flagged targets.

**Assumptions:** `rntl_return_date` is the *actual* return date, so for a rental still open at the cutoff it is future information; we do not use it. We use nothing from the masked test rows (not even their dates) as input.

### Key findings that shaped the model
1. **A car's own past usage is a weak predictor**: the segment average rate (1,680 km MAE) beats the car's own average rate (2,200). Rental cars change customers; the signal is mostly at fleet, segment and location level. The branch pool code carries ~40% of the model's gain.
2. **Busy cars report more often.** Reports are triggered by rental events. In Jul-Nov 2025, cars with <= 10 report days drove a median 29 km/day, cars with 31+ report days 83 km/day (Spearman 0.52). Scoring counts every reading, so busy cars weigh ~3x more. Our first training sampled max 8 targets per car (each car weighted equally), which caused a -400 km bias. **Fix: weight each sampled target by the car's true number of targets.** Bias went from -430 to -194 km.
3. **The remaining error is future rental activity**, which no input contains. Evidence: a linear model is only 5% behind LightGBM, ensembles add 0.3%, more history features add nothing.

## Models and why
Same held-out data for all (20% of vehicles never seen in training, cutoff 2025-07-01, all 34,617 later readings to 2025-11-11). Model comparison run on 50% of the training vehicles.

| Model | Open-weight or closed | Where it runs | MAE km (per-vehicle MAE) | Fit time | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Ridge (scaled numerics + one-hot) | open (scikit-learn) | local CPU | 1,492 (1,562) | 24 s | no, -5% vs GBMs |
| HistGradientBoosting (L1) | open (scikit-learn) | local CPU | 1,434 (1,445) | 32 s | no, slightly worse |
| **LightGBM (L1 on rate, weight = gap)** | open | local CPU | **1,426 (1,438)** | 37 s | **yes**: best, fast, native categoricals |
| LightGBM, 4-seed average | open | local CPU | 1,423 | 69 s | no, +0.2% not worth it |
| Blend Ridge/HistGB/LightGBM (weights fitted on inner slice) | open | local CPU | 1,422 | - | no, +0.3% not worth the complexity |
| Cold-start LightGBM (static attributes + calendar only) | open | local CPU | not validated | - | yes, only for 149 cars (2.3% of test rows) without any history |

No LLM or API is used; prediction for all 163k test rows takes seconds on a laptop.

Hyper-parameter tuning (Optuna TPE, 60 min, 91 trials on a vehicle-disjoint inner slice, then one check on the held-out vehicles): MAE 1,427 → **1,416 km** (-0.8%), MedAE 951 → 939, bias unchanged (-196). Tuned: 50 leaves, strong L2 (47), feature fraction 0.52, heavy smoothing of the pool-code splits (cat_smooth 124, min_data_per_group 609). The final model uses these settings.

## Results
**Forecast (task A)**, held-out vehicles, cutoff 2025-07-01, all target readings (incl. noisy ones), full training set:

| Model | MAE km | MedAE km | WAPE | Bias km |
| --- | --- | --- | --- | --- |
| Carry forward last reading | 5,486 | 4,989 | 1.00 | -5,414 |
| Car's own average rate (≈ `starter.py`) | 2,200 | 1,324 | 0.40 | -320 |
| Segment average rate (best naive) | 1,680 | 1,158 | 0.31 | -547 |
| LightGBM (ours, default settings) | 1,427 | 951 | 0.26 | -194 |
| **LightGBM (ours, tuned)** | **1,416** | **939** | - | -196 |

-16% MAE vs the best naive baseline, -36% vs the starter-style baseline (tuned model). Per-bucket numbers below are for the default-settings run. Per-vehicle MAE (each car counts once): 1,418.

By days since the last known reading (MAE km, ours vs segment baseline): 1-14 d: 417 vs 442; 15-30 d: 805 vs 880; 31-60 d: 1,185 vs 1,341; 61-90 d: 1,566 vs 1,886; 91-150 d: 1,970 vs 2,357. The gain grows with the horizon. In relative terms (WAPE) the error is ~26% of the distance driven at 61-90 days (test median gap: 67 days), and 54% at 1-14 days, where what the car does *right now* dominates.

We report WAPE (sum |error| / sum |true km driven since last reading|) instead of MAPE on the odometer: MAPE on the absolute odometer is tiny and flattering, MAPE on the driven distance explodes for parked cars.

Robustness: TODO, mean ± spread over 3 cutoffs (2024-07-01, two 2025 cutoffs). Earlier runs: the cutoff 2024-07-01 is harder (MAE 1,591) than 2025 (1,386-1,427).

**Test predictions**: all 163,047 masked test rows predicted (`results/test_predictions.csv`), median implied usage 79 km/day (IQR 70-87). No ground truth is available to us, so these are unscored.

**Erroneous mileage detection (task B)**: TODO. Status: the two-sided outlier rule is implemented and used for cleaning (flags ~1% of readings). Planned: rules + residual vs the forecast + synthetic error injection for precision/recall.

**What did not work / limits (honest):**
- More features (pool/region/fleet recent rates, shrunk car rate, on-rent shares) gave no gain at 3 cutoffs.
- Ensembles and denser training cutoffs: within noise.
- Remaining bias -194 km: partly expected (L1 predicts the median of a right-skewed rate), partly the target-count effect.
- Cold-start model not validated; test predictions unscored; no anomaly labels exist, so detection can only be evaluated with injected errors and plausibility review.

## How to run
Python 3.11+. From a fresh clone:
```bash
cd Project_2
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put data_30_percent_sample.parquet and test_data.parquet into Project_2/data/ (git-ignored)
python -m src.run_forecast --frac 1.0 --tag _fix1     # headline validation table -> results/forecast_val_fix1.csv (~15 min)
python -m src.compare_models --frac 0.5              # model comparison -> results/model_comparison.csv (~5 min)
python -m src.hpo --minutes 60 --tag hpo             # optional, hyper-parameter search -> results/hpo/
python -m src.train_final --params results/hpo/hpo_best.json   # final tuned model + test predictions -> results/test_predictions.csv (~12 min)
```
Seeds are fixed (seed 0, same vehicle split in every script). No environment variables or secrets needed. `results/README.md` lists every result file.

## Demo
TODO: `submission/slides.pdf` (max 5 slides).

## Next steps
1. **Anomaly detection on top of the same model**: flag a report when it is far outside the predicted band (using readings before *and* after it), plus physical rules (decrease, impossible km/day, above max mileage). Evaluate with injected errors (dropped digit, rollback, miles/km mix-up).
2. **Typo auto-correction**: for a flagged report, test whether a dropped/swapped digit or a miles/km mix-up explains it and suggest the corrected value.
3. **Uncertainty bands** (quantile models + conformal calibration) per car and date.
4. **Max-mileage crossing date**: predict when each car hits `vhcl_max_mileage`, for defleet planning.
5. **Ask for readings where it matters**: request an odometer reading only from the long-term customers whose cars are most uncertain.
6. **Use SIXT's booking calendar** (future rentals, planned return dates): our analysis shows future rental activity is the missing information, and SIXT already has it.
