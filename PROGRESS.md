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
      period to +24h / +48h / +72h. **Removed in Phase 2b** (superseded by direct
      multi-horizon models); its finding is preserved below.
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
  (RandomForest, best), 0.28 (XGBoost), 0.63 (Ridge). `us_aqi_next` v1 = that
  RandomForest.

Out of scope for this phase: real Hopsworks, LSTM training, dashboard, GitHub
Actions, live weather-forecast fetching.

## Phase 2b – Direct multi-horizon models — done

**Why recursive was dropped.** Walking the 1-hour model forward one step at a
time (old `backtest.py`, 266 test origins) compounded error at every hop, made
worse by having to freeze `pm2_5`/`pm10` (the 1h model only predicts `us_aqi`).
The best recursive model (XGBoost) scored R² 0.584 at +24h but **−0.277 at +48h
and −1.229 at +72h** – i.e. worse than predicting the mean (RandomForest: 0.572 /
−0.293 / −1.252; Ridge: 0.543 / −0.390 / −1.420). `backtest.py` and
`models/backtest_report.json` were removed.

**Replacement.** `training_pipeline/dataset.py` now takes a `horizon_hours`
parameter (target `us_aqi.shift(-h)`, named `us_aqi_next` for h=1 or `us_aqi_h<h>`
otherwise). `training_pipeline/train_multi_horizon.py` builds the dataset at each
of +24h/+48h/+72h, trains Ridge / RandomForest / XGBoost directly on *current*
known features (no recursion, no frozen pm), and registers the best-by-test-RMSE
model as `us_aqi_h24` / `us_aqi_h48` / `us_aqi_h72`. Combined results saved to
`models/training_comparison_multi_horizon.json`.

Direct multi-horizon results (Karachi, best model per horizon, test split):

| horizon | best model     |   RMSE |    MAE |      R² |
| ------- | -------------- | -----: | -----: | ------: |
| +24h    | XGBoost        |  5.292 |  4.296 |  0.7914 |
| +48h    | RandomForest   |  9.943 |  8.379 |  0.2637 |
| +72h    | XGBoost        | 11.905 | 10.436 | −0.0556 |

Direct beats recursive at every horizon (e.g. +72h RMSE 18.2 → 11.9, R² −1.23 →
−0.06). But +48h/+72h were still weak – see Phase 2c, which traced most of that
to a missing feature (no future-weather signal) rather than an inherent limit.

- [x] `dataset.py` – `horizon_hours` param on `build_training_frame` /
      `split_dataset`; `Splits` carries its `target` name; h=1 behaviour and
      `TARGET="us_aqi_next"` unchanged.
- [x] `train_multi_horizon.py` – per-horizon train / compare / register, reusing
      `build_models()` + `train_all()` from `train.py` (which is untouched).
- [x] Deleted `training_pipeline/backtest.py` + `models/backtest_report.json`.
- [x] `tests/smoke_training_pipeline.py` – added horizon-target no-leakage check
      (h=24) and a non-default-horizon split check. All prior checks + the feature
      smoke test still pass.

Out of scope: dashboard, inference wiring, GitHub Actions, live weather-forecast
fetching, LSTM.

## Phase 2c – Weather-at-target-time features — done

**The bug.** `dataset.feature_columns()` only exposed features observed at time
`t`. A model predicting AQI at `t+72h` therefore had *zero* information about the
weather at `t+72h` – wind, humidity and pressure at target time are what actually
drive pollutant dispersion, so the +48h/+72h models were guessing blind on their
most important signal.

**The fix.** `build_training_frame(horizon_hours)` now also adds, for every
`config.WEATHER_VARS` column, a `<var>_target` feature = that column's value at
`t + horizon_hours` (same per-location `groupby(...).shift(-horizon_hours)` used
for the target). These are ordinary inputs (no `feature_columns()` change); the
existing dropna handles their NaN tail. Applied at every horizon incl. h=1 for
interface consistency. Feature count 50 → 56. `train_multi_horizon.py` re-run,
registering **v2** of `us_aqi_h24` / `h48` / `h72` (56 features); `us_aqi_next`
untouched. In a real deployment `<var>_target` would come from a weather
*forecast*; here it is the recorded value (same stand-in already used elsewhere).

Best model per horizon, test split (before = Phase 2b, after = Phase 2c):

| horizon | before RMSE / R²   | after RMSE / R²       | Δ R²   |
| ------- | ------------------ | --------------------- | ------ |
| +24h    | 5.292 / 0.7914     | 4.949 / **0.8176**    | +0.03  |
| +48h    | 9.943 / 0.2637     | 7.910 / **0.5340**    | +0.27  |
| +72h    | 11.905 / −0.0556   | 10.371 / **0.1990**   | +0.25  |

(all three "after" models are XGBoost). Future weather barely moves +24h (as
expected – 24h-ahead weather ≈ now), roughly doubles +48h R², and flips +72h
from worse-than-mean to a real if modest signal. +72h R² ≈ 0.20 is still weak in
absolute terms – 3-day-ahead AQI is genuinely hard, and the 14-day test window
(337 rows) is small – but it is no longer noise.

- [x] `dataset.py` – `<var>_target` weather features added in
      `build_training_frame`; docstring updated.
- [x] `train_multi_horizon.py` re-run – v2 models registered for all 3 horizons.
- [x] `tests/smoke_training_pipeline.py` – added `check_weather_target_no_leakage`
      (time-based lookup, horizon=48). All prior checks + the feature smoke test
      still pass.

Out of scope: dashboard, inference wiring, GitHub Actions, live weather-forecast
fetching, LSTM.

## Still to be placed in a later phase (do not lose track of these)

- **EDA / exploratory notebooks** on the backfilled feature set (`notebooks/`).
- **Model explainability** – SHAP (already a dependency) and LIME (not yet
  installed) for the trained models.
- **Hazardous-AQI alerting** – threshold detection + notification when predicted
  or observed `us_aqi` crosses unhealthy levels.
- **LSTM model** – `training_pipeline/lstm_model.py` is left from the old code and
  is currently unbuildable (it imports the deleted `process.py`). A sequence model
  is deferred; it gets rebuilt against `dataset.py` in a later phase.

## Phase 3 – Live inference pipeline — done

`aqi_predictor/inference_pipeline/predict.py` produces a live multi-horizon US
AQI forecast for a location. `forecast(location_name)`:

1. builds a fully-featured "now" row – `fetch.historical(location, now-4d, today)`
   → `features.build_features()`, take the last row;
2. pulls forward-looking hourly weather – new
   `fetch.forecast_ahead(location, hours_ahead=72)`, which hits the Open-Meteo
   forecast endpoint (`forecast_days` sized to cover the horizon + buffer,
   `past_days=2`);
3. for each horizon in (1, 24, 48, 72): `registry.load_best_model(name)`
   (`us_aqi_next` for +1h, else `us_aqi_h{h}`), builds the input row **strictly
   from `metadata["feature_list"]`** (exact columns and order), overwriting every
   `<var>_target` column with the forecast value at `now + h`; a feature that
   can't be sourced raises rather than silently mismatching;
4. returns `{location, generated_at, current: {time, us_aqi}, forecasts: [{
   horizon_hours, target_time, predicted_us_aqi, model_name, model_version}, …]}`.

CLI: `python -m aqi_predictor.inference_pipeline.predict --location karachi`
(`--json` for raw output).

**How this differs from training.** In `dataset.py` the `<var>_target` columns
are filled with the *recorded* historical weather (a stand-in for "a forecast
would have been available at the time"). Here they come from an actual
forward-looking Open-Meteo forecast — what a deployed system uses. Everything
else in the row (pollutants, time features, lags/rolling on `us_aqi`/`pm*`) is
"as of now" and identical to how the feature store builds it.

**Anchoring "now" (fixed).** Open-Meteo's air-quality endpoint returns data
through the end of the current UTC day (the tail is its own short-range
forecast), so the unfiltered last fetched row can be up to ~23h in the future.
Initially `_now_row()` took that literal last row, which anchored "current AQI"
and every target time to end-of-today UTC — a real correctness issue. Fixed:
`_now_row()` now filters to `time <= now.floor("h")` before taking the last row
(the same guard `fetch.latest_hour` uses), raising a clear `RuntimeError` if
nothing survives. "Current" is now a real recent hour and each `target_time` is
genuinely `current_time + horizon_hours`.

Sample live run (Karachi, 2026-09-01, `generated_at` 08:14 UTC): current us_aqi
61 @ 08:00 → +1h 61.0 @ 09:00 (`us_aqi_next` v1), +24h 68.4 @ 09-02 08:00
(`us_aqi_h24` v2), +48h 73.3 @ 09-03 08:00 (`us_aqi_h48` v2), +72h 71.7 @
09-04 08:00 (`us_aqi_h72` v2) — "current" is within minutes of `generated_at`
and each target time is exactly the horizon out.

- [x] `fetch.forecast_ahead(location, hours_ahead=72)` – forward weather from the
      forecast endpoint, reusing `_get_json` / `_hourly_frame`.
- [x] `inference_pipeline/predict.py` – `forecast()` + `build_input_row()` +
      `--location` CLI.
- [x] `tests/smoke_inference_pipeline.py` – input-row columns == `feature_list`
      exactly / in order; `<var>_target` uses the forecast value for the horizon,
      not the "now" value; missing feature or uncovered target time raises. All
      existing smoke tests still pass.

Out of scope: dashboard/UI, GitHub Actions, hazardous-AQI alerting, LSTM, more
horizons.

## Phase 4 – Streamlit dashboard — done

`aqi_predictor/dashboard/app.py` is a Streamlit app over the live pipeline (no
mock data):

- location dropdown from `config.LOCATIONS` (currently just Karachi);
- `@st.cache_data(ttl=1200)` around `predict.forecast(location_name)` — the
  location is an explicit argument, so the cache keys on it and switching cities
  can never show a stale result for the wrong one;
- current US AQI value + a colour badge from the new shared
  `aqi_predictor/aqi_scale.py` (`aqi_category(value) -> (label, colour_hex)`,
  standard US EPA breakpoints, dependency-free so hazardous-AQI alerting can
  reuse it later);
- a line chart of the last 48h of observed `us_aqi` and a line chart of the
  4-point forecast, plus a forecast table (horizon, target time, value,
  category, model + version);
- the cached call is wrapped in `try/except`; any pipeline failure renders
  `st.error(...)` with the exception message instead of a traceback.

`predict.forecast()` gained a `"recent"` field — `[{time, us_aqi}, …]` for the
last `RECENT_HOURS` (48) of *observed* history, taken from the frame already
fetched for the "now" row (no second live call). `_now_row` became
`_current_and_history`, returning `(now_row, observed_frame)`.

`requirements.txt`: `+ streamlit==1.62.0`.

Verified with `streamlit.testing.v1.AppTest` against the real pipeline: renders
title, badge (AQI 61 → "Moderate", yellow), both line charts, and the 4-row
forecast table with no exception; a forced `predict.forecast` failure renders a
friendly `st.error` and skips the rest of the page.

Run locally: `streamlit run aqi_predictor/dashboard/app.py`

**Deployment is a manual step, outside Claude Code.** After this is pushed,
connect the GitHub repo on share.streamlit.io (main module
`aqi_predictor/dashboard/app.py`), which installs `requirements.txt` and serves
the app. No secrets are needed for the dashboard itself.

- [x] `aqi_predictor/aqi_scale.py` – shared `aqi_category()`, no Streamlit import.
- [x] `predict.forecast()` – added `"recent"` (reuses the already-fetched frame).
- [x] `aqi_predictor/dashboard/app.py` – Streamlit app, cached per location,
      error-guarded.
- [x] `tests/smoke_dashboard.py` – `aqi_category` boundary table + forecast-dict
      shape/timestamp validation. All existing smoke tests still pass.

Out of scope: actual deployment, GitHub Actions, the alerting logic itself (just
the shared `aqi_scale` module), multi-location data.

## Phase 4.5 – Real Hopsworks integration — done

`store.py` and `registry.py` now always call a real Hopsworks project
(`HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT_NAME` from `.env`). **The local
parquet / joblib fallbacks were deleted outright**, not kept alongside — every
public signature is unchanged, so `dataset.py`, `scripts/backfill.py`,
`train.py`, `train_multi_horizon.py` and `predict.py` were untouched (except the
one-line print fix in `train.py`).

- `aqi_predictor/hopsworks_client.py` — lazily-created, module-level-cached
  `hopsworks.login(...)` handle, shared by both modules. `store._project` /
  `registry._project` are thin indirections over it that the smoke tests
  monkeypatch. Two Windows workarounds live here: `cert_folder` is pointed at a
  gitignored repo dir (the client default `/tmp` is un-creatable on Windows),
  and a `<cwd-drive>:\tmp` directory is pre-created (the Kafka storage
  connector's PEM export in `hopsworks_common/client/base.py` hardcodes
  `/tmp` with no override).
- `store.py` — feature group `aqi_features` v1, `primary_key=["location",
  "time"]`, `event_time="time"`, offline-only, **DELTA (not stream),
  `statistics_config=False`**, via `get_or_create_feature_group` (idempotent
  re-runs). DELTA + the `hopsworks[python]` engine writes the feature group
  directly from the client through delta-rs — no server-side Spark job. (The
  default stream/HUDI path *does* run a Spark job; on this setup it needed the
  Kafka `/tmp` hack to even start and then its inline statistics step failed
  with a backend 500 — "Transaction marked for rollback". DELTA sidesteps all of
  that, and statistics are off because the project doesn't use them.)
  `insert_features` strips the tz off `time` before `fg.insert` (Hopsworks'
  offline store rejects tz-aware timestamps — verified against the live
  account); `get_feature_view` reads `fg.read()` back, re-attaches UTC, and
  filters by `[start, end]` + `locations` in pandas exactly as before (empty
  frame if nothing matches, sorted by `(location, time)`).
- `registry.py` — Hopsworks Model Registry. `register_model` uploads a temp dir
  of `model.joblib` + `metadata.json` (full nested metrics, same shape as
  before) via `mr.python.create_model(...).save(...)`, passing only a **flat
  numeric** metrics subset (`test_rmse`/`val_rmse`/…) to `create_model` for the
  UI; returns the Hopsworks-assigned `m.version` (metadata.json's `version` is a
  placeholder, overlaid on read). `list_versions` / `load_model` /
  `load_best_model` download the artifact(s) and read `metadata.json` back, so
  they return exactly what the local implementation did; `load_best_model` picks
  lowest `test_rmse` from the models' attached flat metrics without downloading
  every artifact.
- **Existing feature data was migrated, not re-fetched.**
  `scripts/migrate_to_hopsworks.py` (one-off, not wired into any pipeline) reads
  `data/feature_store/location=*/data.parquet` and pushes it through the new
  `store.insert_features`, then reads it back and checks the row counts /
  time ranges match. `data/feature_store/` is left in place.
- **Existing local models were NOT migrated.** `models/` is left as-is; the 4
  model names (`us_aqi_next`, `us_aqi_h24/h48/h72`) were retrained fresh against
  the Hopsworks-backed registry.
- Tests: `tests/smoke_feature_pipeline.py` / `smoke_training_pipeline.py` no
  longer redirect `FEATURE_STORE_DIR` / `MODELS_DIR` (gone). They monkeypatch
  `store._project` / `registry._project` with small in-memory fakes (a
  feature group with upsert-by-key semantics; a model registry keyed by
  `(name, version)`), so the same round-trip assertions run with no network.

**Dependency fallout (`requirements.txt`).** `hopsworks[python]==5.0.6` pins
`pandas<2.4`, `numpy<2.5`, `protobuf<5`. To make one venv resolve:
- `tensorflow==2.21.0` **dropped** — nothing imports it, and its
  `protobuf>=6.31.1` pin is irreconcilable with Hopsworks. (`torch` stays; it is
  still only used by the deferred, unbuildable `lstm_model.py`.)
- `pandas` 3.0.5 → **2.3.3**, `streamlit` 1.62.0 → **1.59.1**.
- `twofish` (transitive via `pyjks` via `hopsworks`) ships no wheel and there is
  no C compiler on this machine; it is satisfied by a pure-Python import shim
  (`twofish` 0.3.0) that raises if the cipher is ever actually exercised — the
  API-key REST login path never touches it. If this project moves to a machine
  with MSVC build tools, `pip install --force-reinstall --no-binary twofish
  twofish` swaps in the real one.

Out of scope: GitHub Actions/CI, Streamlit Cloud deploy, hazardous-AQI alerting,
LSTM.

## Phase 5 – Automation (GitHub Actions) — not started
