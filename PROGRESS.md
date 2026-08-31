# Progress Tracker

## Phase 0 – Project structure & tooling — done

- [x] `git init`, git config, remote origin -> `https://github.com/alish-lab/AQI-Predictor.git`
- [x] `.gitignore` (`.venv/`, `__pycache__/`, `.env`, `data/`, standard Python)
- [x] `pyproject.toml` so `aqi_predictor` is `pip install -e .`-able
- [x] `requirements.txt` frozen from the working venv (source of truth for pins)
- [x] `.env.example`
- [x] Reorg into `aqi_predictor/{feature_pipeline,training_pipeline,inference_pipeline,dashboard}/`, `scripts/`, `notebooks/`, `tests/`
- [x] `README.md`, `PROGRESS.md`
- [x] Initial commit + push to `origin/main`

Out of scope for that phase: Hopsworks/Streamlit/LIME/statsmodels installs, feature
engineering, fixing the data-leakage / hardcoded-date issues, GitHub Actions.

## Phase 1 – Feature pipeline (rebuild + historical backfill) — done

- [x] `aqi_predictor/config.py` – `LOCATIONS` (multi-location schema from day one),
      shared paths/constants, `.env` loading via `python-dotenv`
- [x] `LATITUDE`/`LONGITUDE` moved out of `.env`/`.env.example` into `config.LOCATIONS`
      (they are not secrets); `.env.example` now only `HOPSWORKS_API_KEY` /
      `HOPSWORKS_PROJECT_NAME`
- [x] `feature_pipeline/fetch.py` – `historical()` and `latest_hour()`:
  - [x] `end_date` computed dynamically (UTC "today"), not hardcoded
  - [x] air-quality endpoint (pollutants + `us_aqi`) **and** weather via the
        historical-weather archive endpoint, merged on `(location, time)`
  - [x] weather vars: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`,
        `wind_direction_10m`, `surface_pressure`, `precipitation`
  - [x] archive endpoint lags ~1 day; the gap to "now" is filled from the
        forecast endpoint's `past_days` window (per decision in the phase brief)
  - [x] `carbon_dioxide` not fetched (was 100% null in the old CSV)
- [x] `feature_pipeline/features.py` – time features (hour / dow / month /
      is_weekend / cyclical sin-cos for hour & month), AQI change-rate (1h, 24h),
      lags (t-1/2/3 on `us_aqi`, `pm2_5`, `pm10`), rolling mean & std (3h/6h/24h
      on the same three). All backward-looking – no row sees its own future.
- [x] Missing-data handling: linear-interpolate gaps <= 3h; drop rows still
      missing `us_aqi` / `pm2_5` / `pm10`; interpolated vs dropped counts logged
- [x] `feature_pipeline/store.py` – local partitioned-parquet feature store under
      `data/feature_store/`, Hopsworks-shaped interface:
      `insert_features(df)` (upsert on `(location, time)`, idempotent) and
      `get_feature_view(start, end, locations=None)`
- [x] `scripts/backfill.py` – fetch (monthly chunks) -> features -> store, for
      every location, 2024-07-29 -> today
- [x] Data-quality report printed and saved to
      `data/feature_store/backfill_report.json` (row count, date coverage,
      % missing per column before/after handling, rows dropped, cells interpolated)
- [x] `data/raw/` and `data/feature_store/` gitignored (`.gitignore` `data/` rule)
- [x] `tests/smoke_feature_pipeline.py` – offline invariant checks (no-leakage,
      missing-data handling, store round-trip)
- [x] Backfill run: Karachi, 18 336 hourly rows, 2024-07-29 -> 2026-08-31,
      0 gaps / 0 interpolated / 0 dropped (Open-Meteo reanalysis is gap-free)

Notes / deferred:
- Real Hopsworks integration is **not** done – `store.py` is a local fallback with
  the same call signatures; a later phase swaps the internals.

Out of scope for this phase: real Hopsworks integration, training-pipeline
changes, dashboard, GitHub Actions/CI.

## Phase 2 – Training pipeline (rebuild) — done

- [x] `training_pipeline/dataset.py` – reads the full feature store via
      `get_feature_view()`, adds per-location `us_aqi_next = us_aqi.shift(-1)`,
      drops each location's last row + first ~24h (undefined long-window features).
      Time-ordered per-location split: last 14 days = test, prior 14 days = val,
      rest = train. No shuffling.
- [x] `training_pipeline/train.py` – trains Ridge (with a **train-fit**
      `StandardScaler` in a Pipeline, fixing the old scaler-leakage bug),
      RandomForest, XGBoost to predict `us_aqi_next`; evaluates all three on val
      and test with RMSE / MAE / R²; prints + saves a comparison table
      (`models/training_comparison.json`). The old "accuracy" metric is gone.
- [x] `training_pipeline/metrics.py` – shared RMSE / MAE / R² (no "accuracy").
- [x] `training_pipeline/backtest.py` – recursive walk-forward over the test
      period to +24h / +48h / +72h, per model, reporting RMSE / MAE / R² at each
      horizon (`models/backtest_report.json`). Future weather = real historical
      rows (stand-in for a real forecast); `us_aqi`-derived features follow the
      predicted trajectory; `pm2_5`/`pm10` (+ other pollutants) and their
      lag/rolling features are frozen at the forecast origin – documented
      simplification since the model only predicts `us_aqi`.
- [x] `training_pipeline/registry.py` – local model registry under `models/`
      (git-ignored), Hopsworks-shaped: `register_model(name, model, metrics,
      feature_list) -> version` and `load_best_model(name) -> (model, metadata)`
      (best = lowest test RMSE). joblib + metadata JSON per version.
- [x] `train.py` registers the best model (by test RMSE) as `us_aqi_next`.
- [x] Deleted superseded files: `feature_pipeline/api_call.py`,
      `training_pipeline/process.py`, `training_pipeline/xg_model.py`.
      `training_pipeline/lstm_model.py` left in place (deferred – see above).
- [x] `tests/smoke_training_pipeline.py` – target no-leakage, split non-overlap,
      registry round-trip. Existing `tests/smoke_feature_pipeline.py` still passes.
- [x] `.gitignore` – added `models/`.

Run so far (Karachi, 2024-07-29 → 2026-08-31):
- 1-hour models are near-perfect (persistence dominates): test RMSE ≈ 0.16
  (RandomForest, best), 0.28 (XGBoost), 0.63 (Ridge).
- Recursive multi-horizon degrades as expected: at +24h RMSE ≈ 7.6 / R² ≈ 0.58;
  at +48h and +72h R² goes negative – a dedicated multi-step approach is future
  work.

Out of scope for this phase: real Hopsworks, LSTM training, dashboard, GitHub
Actions, live weather-forecast fetching.

## Still to be placed in a later phase (do not lose track of these)

- **EDA / exploratory notebooks** on the backfilled feature set (`notebooks/`).
- **Model explainability** – SHAP (already a dependency) and LIME (not yet
  installed) for the trained models.
- **Hazardous-AQI alerting** – threshold detection + notification when predicted
  or observed `us_aqi` crosses unhealthy levels.
- **LSTM model** – `training_pipeline/lstm_model.py` is left from the old code and
  is currently unbuildable (it imports the deleted `process.py`). A sequence model
  is deferred; it gets rebuilt against `dataset.py` in a later phase.

## Phase 3 – Inference pipeline & fixes — not started

## Phase 4 – Dashboard — not started

## Phase 5 – Automation (GitHub Actions) — not started
