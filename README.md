# AQI Predictor

Air-quality (US AQI) forecasting for Karachi using hourly Open-Meteo data. The
project is organised as a set of pipelines:

- **feature_pipeline** – fetch air-quality + weather from Open-Meteo, engineer
  features, and store them (`fetch.py`, `features.py`, `store.py`).
- **training_pipeline** – build the supervised dataset, train and compare models
  for next-hour `us_aqi`, backtest recursively, and register the best
  (`dataset.py`, `train.py`, `backtest.py`, `registry.py`).
- **inference_pipeline** – generate predictions (to be built).
- **dashboard** – visualise forecasts (to be built).

The schema is multi-location from the start (every row carries a `location`
key), though only Karachi is active. See [PROGRESS.md](PROGRESS.md) for the phase
tracker.

## Project layout

```
aqi_predictor/
  config.py            LOCATIONS, paths, constants, .env loading
  feature_pipeline/
    fetch.py           Open-Meteo air-quality + weather, merged on (location, time)
    features.py        engineered features + missing-data handling
    store.py           local parquet feature store (Hopsworks-shaped interface)
  training_pipeline/
    dataset.py         supervised frame (us_aqi_next target) + time-ordered split
    train.py           Ridge / RandomForest / XGBoost, comparison, register best
    backtest.py        recursive +24h/+48h/+72h walk-forward over the test period
    registry.py        local model registry (Hopsworks-shaped interface)
    lstm_model.py       old code, deferred (currently unbuildable)
  inference_pipeline/  (placeholder)
  dashboard/           (placeholder)
scripts/
  backfill.py          historical backfill: fetch -> features -> store
tests/
  smoke_feature_pipeline.py    offline invariant checks (run with plain python)
  smoke_training_pipeline.py   offline invariant checks (run with plain python)
data/                  local data, git-ignored (raw pulls + feature store)
models/                local model registry + reports, git-ignored
```

## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
source .venv/bin/activate          # macOS / Linux

# 2. Install pinned dependencies (source of truth) + the package
pip install -r requirements.txt
pip install -e .

# 3. Configure secrets
cp .env.example .env               # then edit .env
```

## Environment variables

Only secrets live in `.env` (see [.env.example](.env.example)). Coordinates and
locations are code, in `aqi_predictor/config.py`.

| Variable                 | Purpose                          |
| ------------------------ | -------------------------------- |
| `HOPSWORKS_API_KEY`      | Hopsworks feature store API key  |
| `HOPSWORKS_PROJECT_NAME` | Hopsworks project name           |

## Usage

```bash
# Backfill the full engineered feature history into data/feature_store/
python scripts/backfill.py

# ...for a shorter range or a single location
python scripts/backfill.py --start 2026-01-01 --location karachi
```

```python
from aqi_predictor.feature_pipeline import store

df = store.get_feature_view("2026-01-01", "2026-06-30")   # read a slice back
```

A data-quality report is printed after each backfill and saved to
`data/feature_store/backfill_report.json`.

```bash
# Train + compare models, register the best by test RMSE
python -m aqi_predictor.training_pipeline.train

# Recursive multi-horizon backtest (+24h / +48h / +72h)
python -m aqi_predictor.training_pipeline.backtest
```

```python
from aqi_predictor.training_pipeline import registry

model, metadata = registry.load_best_model("us_aqi_next")
```

Model artifacts, `training_comparison.json` and `backtest_report.json` land in
`models/` (git-ignored).

