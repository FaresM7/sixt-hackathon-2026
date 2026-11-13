# results/: what each file is

All validation numbers: held-out 20% of vehicles (seed 0), cutoff 2025-07-01, targets to 2025-11-11,
34,617 target readings, unless stated. Columns: MAE / MedAE / WAPE / bias (km), `MAE_vw` = each
vehicle counts once. Rows `targets=all` include flagged readings (as the hidden truth would);
`ok_only` excludes them.

| file | setup | LightGBM MAE / bias |
|---|---|---|
| `forecast_val_tuned.csv` | **headline**: as fix1 + tuned params (`--params results/hpo/hpo_best.json`) | 1,416 / -196 |
| `forecast_val_tuned_c2407.csv`, `forecast_val_tuned_c2501.csv` | robustness: tuned model at cutoffs 2024-07-01 and 2025-01-01 | 1,560 / 1,379 |
| `forecast_seg_tuned.csv` | MAE by segment (rental product, on rent, fleet status, history length, last-reading age) | |
| `bands_tuned.csv` | 80% prediction bands: coverage raw / conformal, width, per horizon | coverage 0.82 |
| `coldstart_val.csv` | `src.eval_coldstart`: cold-start model vs new-car baselines | 2,465 vs 2,763 |
| `plots/forecast_by_horizon.png`, `plots/anomaly_vs_starter.png` | slide charts (`src.make_charts`) | |
| `forecast_val_fix1.csv` | target-count weighting, no return-date features, base features, default params | 1,427 / -194 |
| `forecast_val_nocap.csv` | all targets per vehicle, still with return-date features | 1,431 / -271 |
| `forecast_val_nocap_noret.csv` | all targets, return-date features dropped (dropping them costs nothing) | 1,429 / -268 |
| `forecast_val_c24_old/new.csv` | cutoff 2024-07-01; old = base features, new = + group-rate features | 1,591 / 1,586 |
| `forecast_val_c25a_old/new.csv` | second 2025 cutoff; old vs new features | 1,386 / 1,385 |
| `model_comparison.csv` | `src.compare_models` (50% of training vehicles): Ridge, HistGB, LightGBM, seed bag, ensembles | see file |
| `anomaly_eval.csv`, `anomaly_correction.csv`, `anomaly_flags_top.csv`, `plots/anomaly_examples.png` | task B (`src.anomalies`): injection eval, typo-correction accuracy, top real flags, examples | |
| `hpo/` | `src.hpo` study (`<tag>.db`), trials, best params, tuned-vs-default held-out check | |
| `test_predictions.csv` | `src.train_final`: predictions for all 163,047 masked test rows (git-ignored, unscored) | |
| `archive/` | superseded runs (capped targets without re-weighting and/or leaky return-date features); kept for the history only | 1,438-1,487 |
