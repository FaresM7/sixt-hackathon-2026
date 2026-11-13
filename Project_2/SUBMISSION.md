# Submission: Project 2, The golden butterfly

## Authors
**Team name:** The golden butterfly

**Project:** Project 2: Mileage Forecasting & Erroneous Mileage Detection

## Problem and approach
SIXT values and defleets cars on mileage it often does not know yet: long-term rentals report weeks late (in the test file the target date is a median 67 days after the last reading), and some reports are wrong.

**One engine for both tasks.** We predict a car's **km per day** from what is known at the cutoff (recent usage, rental state, vehicle type, branch pool, season): `km(t) = last reading + predicted km/day x days`. LightGBM is trained by replaying the past (a pretend "today" every 30 days since 2023, real later readings as targets) and tested on vehicles it never saw. The same expectations drive the anomaly detection: physical rules, a check against the readings before and after each report, and a real-time mode using the forecast.

**Key finding:** busy cars report more often (29 vs 83 km/day for cars with <= 10 vs 31+ reports), so every reading is not an equal sample. Weighting training targets accordingly removed a -400 km bias.

**Assumptions / data rules (confirmed with the project owner):** `rntl_return_date` is the actual return date, so it is not used as a feature (future information); the test file is never used for training, nor anything from its masked rows.

## Models and why
Same held-out data for all: 20% of vehicles never seen in training, cutoff 2025-07-01, all 34,617 later readings (model comparison on 50% of the training vehicles).

| Model | Open-weight or closed | Where it runs | MAE km | Cost and time | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Ridge regression | open (scikit-learn) | local CPU | 1,492 | free; 24 s fit | no: 5% worse |
| HistGradientBoosting | open (scikit-learn) | local CPU | 1,434 | free; 32 s fit | no: slightly worse |
| **LightGBM, km/day target** | open | local CPU | **1,426** | free; 37 s fit, < 1 ms per prediction | **yes**: best and fast |
| Seed average / blend of the above | open | local CPU | 1,422-1,423 | 2x time | no: +0.3%, not worth it |

Final model: LightGBM with tuned settings (Optuna, 91 trials on a separate inner slice), trained on all training vehicles: **MAE 1,416 km**. A small extra model covers the 149 test cars with no history (2.3% of rows). No LLM or API.

## Results
**Forecast** (held-out vehicles, MAE in km; full tables in `DETAILS.md`):

| | Ours | Segment average (best naive) | Car's own rate (`starter.py`) | Carry forward |
| --- | --- | --- | --- | --- |
| Headline period (Jul-Nov 2025) | **1,416** | 1,680 | 2,200 | 5,486 |
| Mean over 3 validation periods | **1,452** | 1,694 | 2,154 | 5,247 |

- -16% vs the best naive baseline, -36% vs the starter baseline; the gain holds in all 3 periods and grows with the horizon.
- Last reading older than 28 days (monthly reporting): 1,722 vs 3,291 (-48%).
- 80% uncertainty bands cover 82% of true values. Summed per fleet, the km error is 3.6%.

**Erroneous mileage detection** (no labels: 3,050 known errors planted into held-out vehicles, 7 types):

| | Readings flagged | Errors >= 1,000 km caught | Flags that are planted errors (lower bound) |
| --- | --- | --- | --- |
| `starter.py` rules | 2.6% | 93% | 21% |
| **Ours** | **0.5%** | **92%** | **57%** |

Same catch rate with 5x fewer alarms. Suggested corrections are right for 97-98% of swapped digits and miles-as-km. On the organisers' test data we flag 1,018 readings, e.g. 551,098 km between 55,074 and 55,412, suggested fix 55,109.

**Limits:** the remaining error is future rental activity (a linear model is only 5% behind; ensembles add 0.3%); stuck odometers cannot be detected from mileage alone; planted errors are an optimistic proxy; test predictions are unscored.

## How to run
**Data is not included in this repo.** The organisers' `data/` folder was removed for confidentiality (it contained a credential that does not belong in a public repo). Place the two files provided by the organisers into `Project_2/data/` (git-ignored) before running:
`Project_2/data/data_30_percent_sample.parquet` and `Project_2/data/test_data.parquet`.

Python 3.11+, no environment variables:
```bash
cd Project_2
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train_final --params results/hpo/hpo_best.json   # forecast + predictions for all test rows (~12 min)
python -m src.anomalies                                         # anomaly flags + evaluation (~2 min)
```
All result tables are committed in `results/` (index: `results/README.md`); the commands to regenerate each of them are in `DETAILS.md`.

## Demo
[`submission/slides.pdf`](submission/slides.pdf) (5 slides).

## Next steps
- **Use SIXT's booking calendar**: future rental activity is the missing information, and SIXT already has it.
- **Review queue**: flags + reason + suggested fix go to a person; their decisions become labels for a supervised detector.
- **Ask for readings where it matters**: a check-in reading from the 20% most uncertain cars removes 8.4% of the error vs 5.8% for random cars.
- **Real-time scoring** of every incoming report with the forecast model.
