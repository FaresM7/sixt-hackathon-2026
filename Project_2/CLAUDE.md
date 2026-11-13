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
- **Hypothesis (verify first!)**: for `OnRent` rows, `vhcl_current_mileage` = odometer at rental **handover** (start); the `NotOnRent` row on the return date = odometer at **return**. Consistently, return km of rental i == start km of rental i+1. So trip distance = return km - start km, and nowcast of an ongoing rental = start km + rate * days since handover. Confirm with the project owners (Julian/Ernest).
- `NotOnRent` rows have null handover/return dates. `vhcl_max_mileage` can change over time (25000 -> 24000). `status_code` 3 (defleet) has `uti_plan` = 0. First reading of a new car is ~5 km.
- Example vehicle: ~70 km/day average, only ShortTerm rentals. The full data should contain long-term rentals with sparse reporting.

## Conventions
- Python 3.11+, deps in `requirements.txt` (numpy, pandas, scikit-learn, matplotlib; add lightgbm if used). Root `.venv` exists but has no packages yet.
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
