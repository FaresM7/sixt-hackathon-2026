# Submission: Project 2, The golden butterfly

> Status 13 Nov, 14:45. TODO: slides.

## Authors
**Team name:** The golden butterfly

**Project:** Project 2: Mileage Forecasting & Erroneous Mileage Detection

## Problem and approach
SIXT needs a vehicle's current and future odometer reading for valuation and defleet decisions, but readings arrive irregularly (long-term rentals report up to weeks late). The test file reflects this: the last known reading is on 2026-07-01, and we must predict the km at ~20 given dates per car, 1-147 days later (median 67), with no rental status known at those dates.

**Main idea: predict a usage rate, not an odometer value.**
`km(t) = last reading + predicted km/day x days since that reading`.
A gradient-boosted model (LightGBM) learns the km/day from what is known at the cutoff: the car's recent usage, its rental state, its type and its branch/pool. The loss is L1 on the rate weighted by the gap in days, which equals the absolute error in km, so we optimise MAE directly. Predictions are clipped to [0, p99.9 rate] and made monotone per car.

**How we train it (mimicking the test):** pick many pseudo-cutoffs in the past (every 30 days since 2023-03). For each, freeze what was known at that date, and use the car's real later readings (up to 150 days ahead) as targets. One model covers all horizons; the gap is a feature.

**Data cleaning:** one reading per car-day (the max); a reading is flagged as an outlier if it breaks monotonicity against the median of its 3 previous / 3 next readings by > 100 km, or is absurd (>= 900k km, > 3x max mileage); 1% of readings are flagged. The clean series is forced monotone. Training targets use clean readings only; we report validation both with and without flagged targets.

**Data rules (confirmed with the project owner):** `rntl_return_date` is the *actual* return date, so for a rental still open at the cutoff it is future information; we do not use it as a feature. The test file is for testing only: nothing from it is used for training, and nothing from its masked rows (not even their dates) is used as input. The test file's known history (up to 2026-07-01) is only used as the starting point of its own predictions.

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
| Cold-start LightGBM (static attributes + calendar only) | open | local CPU | 2,465 on 416 held-out new cars (baseline: median km of new cars at that horizon 2,763) | - | yes, only for 149 cars (2.3% of test rows) without any history |

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
| **LightGBM (ours, tuned)** | **1,416** | **939** | **0.26** | -196 |

-16% MAE vs the best naive baseline, -36% vs the starter-style baseline, -74% vs carry-forward (tuned model, `results/forecast_val_tuned.csv`). Per-vehicle MAE (each car counts once): 1,402 vs 1,793 for the segment baseline.

By days since the last known reading (MAE km, tuned model vs segment baseline vs starter-style): 1-14 d: 415 / 442 / 478; 15-30 d: 801 / 880 / 990; 31-60 d: 1,176 / 1,341 / 1,597; 61-90 d: 1,551 / 1,886 / 2,322; 91-150 d: 1,959 / 2,357 / 3,395. The gain grows with the horizon. In relative terms (WAPE) the error is ~26% of the distance driven at 61-90 days (test median gap: 67 days), and 54% at 1-14 days, where what the car does *right now* dominates.

We report WAPE (sum |error| / sum |true km driven since last reading|) instead of MAPE on the odometer: MAPE on the absolute odometer is tiny and flattering, MAPE on the driven distance explodes for parked cars.

**Robustness: three validation cutoffs** (each: different held-out period, same 20% held-out vehicles, tuned model):

| Cutoff (window) | LightGBM | Segment baseline | Starter-style | Gain vs segment / starter | Bias (ours) |
| --- | --- | --- | --- | --- | --- |
| 2024-07-01 (to 11-11) | 1,560 | 1,837 | 2,425 | -15% / -36% | -453 |
| 2025-01-01 (to 05-14) | 1,379 | 1,564 | 1,838 | -12% / -25% | +1 |
| 2025-07-01 (to 11-11), headline | 1,416 | 1,680 | 2,200 | -16% / -36% | -196 |
| **Mean** | **1,452** | 1,694 | 2,154 | **-14% / -33%** | |

The gain holds in every period and season. The bias varies with the period (-453 to +1 km): it follows year-to-year changes in fleet usage that no feature captures.

**By segment** (headline cutoff, state at the last known reading; MAE km, ours vs segment baseline):
- Long-term rental: 2,052 vs 2,278 (-10%); short-term: 1,336 vs 1,606 (-17%). Long-term cars are harder (they report rarely and drive more), but we still beat the baseline.
- Last reading older than 28 days (the "monthly reporting" case): 1,722 vs 3,291 (**-48%**, 609 readings). Defleet cars (status 3): 1,024 vs 2,144 (-52%, small sample: 120 readings): the model learns that they stop driving.
- Short history (< 90 days): 1,417 vs 1,752 (-19%; the car's own rate alone: 3,038).

**Uncertainty: 80% prediction bands.** Quantile LightGBMs (10th/90th percentile of the rate) plus a split-conformal correction fitted on the inner slice. Measured coverage on the held-out vehicles: **82%** overall (80-84% in every horizon bucket; 77% before the correction). Median band width: 1,190 km at 1-14 days, 6,770 km at 91-150 days. Every forecast can therefore be shipped with a range, e.g. "23,400 km, 80% sure between 21,300 and 25,900".


**Business value simulations** (`python -m src.business_sim`, on the held-out predictions, no new training):
- **Fleet-level error is small.** Valuation and defleet planning look at totals. The error relative to km driven is 26% for a single reading, but errors cancel when summed: **6.5% per branch pool, 5.3% per region, 3.6% for the whole held-out fleet**.
- **Where to ask customers for a reading.** A fresh reading on the cutoff day barely helps (removes 5% of the total error even if *every* car reports): most cars' last reading is < 1 week old, so the error is about *future* usage. A **check-in reading 60 days into the window** removes 28% of the total error if every car reports. Asking only the 20% of cars with the widest 80% band removes 8.4% (vs 5.8% for 20% random cars; 10%: 5.1% vs 2.8%): the bands identify where a reading is worth ~1.5-1.8x more. Simplification: after the reading, the same predicted km/day is applied (no re-forecast).

**Test predictions**: all 163,047 masked test rows predicted (`results/test_predictions.csv`), median implied usage 79 km/day (IQR 70-87). No ground truth is available to us, so these are unscored.

### Erroneous mileage detection (task B), `src/anomalies.py`
No labels exist, so we built label-free detectors and evaluated them by **injecting known errors** into 1% of the clean readings of the 5,526 held-out vehicles (3,050 errors in 354k readings): dropped digit, extra digit, swapped digits, rollback (-1k to -20k km), upward offset (+2k to +20k km), stuck odometer (previous value repeated), miles reported as km.

Detectors (one engine, two modes + rules):
- **Rules** (physical, "certain"): absurd value, reset to ~0, mileage drop, implied km/day above the p99.9 of real intervals (3,018 km/day). Drop/rate rules only blame the reading that is itself out of line, not the correct one after it.
- **Two-sided score** (past reports, "likely"): an odometer only goes up, so a reading must lie between the median of its 5 previous and 5 next readings. Score = km outside that range, computed in two passes (suspects from pass 1 are excluded from the reference), and a reading that is consistent with one side is not blamed (protects the correct reading next to a block of bad ones).
- **Forward score** (real-time mode, new report with nothing after it yet): the **task-A LightGBM forecast** from the history before the report; score = |reported - expected| / typical error at that gap.

Results (alarm budget = share of non-injected readings flagged; precision is a lower bound because flags on real errors count as false):

| Detector | Readings flagged | Recall, all injected | Recall, errors >= 1,000 km | Precision (lower bound) |
| --- | --- | --- | --- | --- |
| `starter.py` rules (baseline: drop or > 1,000 km/day) | 2.6% | 0.79 | 0.93 | 0.21 |
| Our rules only | 0.2% | 0.66 | 0.81 | 0.71 |
| **Our rules + two-sided score** | **0.5%** | **0.75** | **0.92** | **0.57** |
| Real-time: forecast-model score (validation window) | 1.0% | 0.58 | 0.75 | 0.35 |
| Same window: two-sided score | 0.7% | 0.72 | 0.93 | 0.49 |

=> Same recall of large errors as the starter rules with **5x fewer alarms** and ~3x the precision. Recall per type (rules + two-sided): extra digit 100%, offset up 97%, rollback 95%, dropped digit 93%, miles-as-km 82%, swapped digits 33% (most swaps change the value by < 100 km and are harmless), **stuck odometer 0%** (a repeated value is indistinguishable from a parked car without telematics). The real-time forecast score is weaker than the two-sided one (no "after" readings) but still catches 75% of large errors at a 1% alarm rate; it is the mode for checking a report the moment it arrives.

**Typo auto-correction** (flagged injected typos): for a flagged reading we try digit insertions/deletions, adjacent swaps and x1.609, and suggest the candidate inside the expected range closest to the interpolated value. Exact recovery: miles-as-km 98%, swapped digits 97%, extra digit 54% (87% within 100 km), dropped digit 37% (67% within 100 km; the missing digit is often ambiguous).

**On the real data** (all 1.78M daily readings, no injection): 8,642 readings flagged (0.49%): 4,147 certain (rules), 4,495 likely (two-sided); 32% come with a suggested correction. Top 500 with reasons in `results/anomaly_flags_top.csv`, examples plotted in `results/plots/anomaly_examples.png`.

**On the organisers' test file** (known history up to 2026-07-01, `python -m src.anomalies --flag_file data/test_data.parquet`): 1,018 of 330,598 readings flagged (0.31%) in 644 of 8,105 vehicles (586 certain, 432 likely), 38% with a suggested correction. The top flags are textbook typos with an exact fix, e.g. 551,098 km between readings of 55,074 and 55,412 -> suggested 55,109; 400,695 between 40,549 and 40,696 -> 40,695; 386,199 between 38,460 and 38,619 -> 38,619. Full list with vehicle ids and dates: `results/anomaly_flags_test_data.csv`.

**What did not work / limits (honest):**
- More features (pool/region/fleet recent rates, shrunk car rate, on-rent shares) gave no gain at 3 cutoffs.
- Ensembles and denser training cutoffs: within noise.
- Remaining bias -194 km: partly expected (L1 predicts the median of a right-skewed rate), partly the target-count effect.
- Cold-start model: -11% MAE vs its baseline and almost no bias, but the error is large relative to the ~3,400 km such cars have driven (we don't know when a new car enters the fleet). Test predictions unscored; no anomaly labels exist, so detection can only be evaluated with injected errors and plausibility review. Injected errors are an optimistic proxy for real ones.
- Stuck odometers are not detectable from mileage reports alone; long blocks of wrong readings (6+ in a row) can still confuse the two-sided score (rare, visible in the examples).

## How to run
Python 3.11+. From a fresh clone (the two parquet files are already in `Project_2/data/`):
```bash
cd Project_2
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Main pipeline (2 commands, ~15 min)**, uses the committed tuned parameters, no search needed:
```bash
python -m src.train_final --params results/hpo/hpo_best.json   # task A: final model + predictions for all masked test rows -> results/test_predictions.csv (~12 min)
python -m src.anomalies                                         # task B: flags on the real data + injection evaluation + plots (~2 min)
```

**Reproduce the evaluation tables in this document (optional).** All result files are already committed in `results/` (index: `results/README.md`); these commands regenerate them:
```bash
python -m src.run_forecast --frac 1.0 --params results/hpo/hpo_best.json --bands --tag _tuned   # headline + per-horizon + per-segment tables, 80% bands (~20 min)
python -m src.run_forecast --frac 1.0 --params results/hpo/hpo_best.json --cutoff 2024-07-01 --val_end 2024-11-11 --tag _tuned_c2407   # robustness, 2nd cutoff (~10 min)
python -m src.run_forecast --frac 1.0 --params results/hpo/hpo_best.json --cutoff 2025-01-01 --val_end 2025-05-14 --tag _tuned_c2501   # robustness, 3rd cutoff (~10 min)
python -m src.compare_models --frac 0.5     # model-family comparison and ensembles (~5 min)
python -m src.eval_coldstart                # cold-start model validation (~1 min)
python -m src.business_sim                  # fleet-level error + reading-request simulation (needs the --bands run; ~1 min)
python -m src.make_charts                   # slide charts -> results/plots/
python -m src.hpo --minutes 60 --tag hpo    # hyper-parameter search; rewrites results/hpo/hpo_best.json (~75 min)
```
Seeds are fixed (seed 0, same vehicle split in every script). No environment variables or secrets needed.

## Demo
TODO: `submission/slides.pdf` (max 5 slides).

## Next steps
1. **Use SIXT's booking calendar** (future rentals, planned return dates, contract ends): our analysis shows that future rental activity is the missing information, and SIXT already has it.
2. **Review queue with feedback**: flags + reason + suggested fix go to a human; their decisions become labels to train a supervised detector and tune thresholds.
3. **Ask for readings where it matters**: use the band width to request an odometer reading only from the long-term customers whose cars are most uncertain.
4. **Score every incoming report in real time** with the forecast model (the forward score), as a nightly or streaming job.
5. **Max-mileage crossing date**: predict when each car hits `vhcl_max_mileage` (with its band), for defleet planning.
