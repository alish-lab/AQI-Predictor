# Progress Tracker

## Phase 0 – Project structure & tooling — done

- [x] `git init`, git config, remote origin -> `https://github.com/alish-lab/AQI-Predictor.git`
- [x] `.gitignore` (`.venv/`, `__pycache__/`, `.env`, `data/raw_aqi.csv`, standard Python)
- [x] `pyproject.toml` so `aqi_predictor` is `pip install -e .`-able
- [x] `requirements.txt` frozen from the working venv (source of truth for pins)
- [x] `.env.example` (`HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`, `LATITUDE`, `LONGITUDE`)
- [x] Reorg into `aqi_predictor/{feature_pipeline,training_pipeline,inference_pipeline,dashboard}/`, `scripts/`, `notebooks/`, `tests/`
- [x] Relocate Open-Meteo fetch logic from `api_call.py` into `feature_pipeline/` (moved, imports fixed, not rewritten)
- [x] Delete `debug_openmeteo.py`
- [x] `README.md`, `PROGRESS.md`
- [x] Initial commit + push to `origin/main`

Out of scope for this phase: Hopsworks/Streamlit/LIME/statsmodels installs, feature
engineering, fixing the data-leakage / hardcoded-date issues, GitHub Actions.

## Phase 1 – Feature pipeline (rebuild) — not started

## Phase 2 – Training pipeline (rebuild) — not started

## Phase 3 – Inference pipeline & fixes — not started

## Phase 4 – Dashboard — not started

## Phase 5 – Automation (GitHub Actions) — not started
