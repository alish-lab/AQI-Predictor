# AQI Predictor

Air-quality (US AQI) forecasting for Karachi using hourly Open-Meteo air-quality
data. The project is organised as a set of pipelines:

- **feature_pipeline** – fetch raw air-quality data from Open-Meteo.
- **training_pipeline** – build sequences and train models (LSTM, XGBoost).
- **inference_pipeline** – generate predictions (to be built).
- **dashboard** – visualise forecasts (to be built).

This repo is in an early restructuring phase. See [PROGRESS.md](PROGRESS.md) for
the phase tracker. Data, features, and models are carried over as-is and will be
rebuilt in later phases.

## Project layout

```
aqi_predictor/
  feature_pipeline/     Open-Meteo fetch logic (api_call.py)
  training_pipeline/    process.py, lstm_model.py, xg_model.py
  inference_pipeline/   (placeholder)
  dashboard/            (placeholder)
scripts/                entry-point / one-off scripts
notebooks/              exploratory notebooks
tests/                  test suite
data/                   local data (raw_aqi.csv is git-ignored)
```

## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install pinned dependencies (source of truth)
pip install -r requirements.txt

# 3. Install the package in editable mode
pip install -e .

# 4. Configure environment
cp .env.example .env   # then edit .env with real values
```

## Environment variables

See [.env.example](.env.example):

| Variable                 | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| `HOPSWORKS_API_KEY`      | Hopsworks feature store API key           |
| `HOPSWORKS_PROJECT_NAME` | Hopsworks project name                    |
| `LATITUDE`               | Latitude for Open-Meteo queries           |
| `LONGITUDE`              | Longitude for Open-Meteo queries          |

## Usage

```bash
# Fetch raw air-quality data -> data/raw_aqi.csv
python -m aqi_predictor.feature_pipeline.api_call
```
