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

**Deployed on Streamlit Community Cloud.** Connected the GitHub repo at
share.streamlit.io (main module `aqi_predictor/dashboard/app.py`). Three real
issues surfaced during deployment, none of them code bugs in the usual sense:

- **Python version**: the app's Python-version setting defaulted to 3.14,
  which is *too new* — `hopsworks==5.0.6` requires `<3.14,>=3.10`, while
  `xgboost==3.3.0` requires `>=3.12` (same constraint that broke the GitHub
  Actions workflows in Phase 5). The only versions satisfying both are 3.12 or
  3.13 — set explicitly in the app's settings, not left on the platform
  default.
- **`ModuleNotFoundError: No module named 'aqi_predictor'`** despite a clean
  dependency install: Streamlit Cloud only runs `pip install -r
  requirements.txt`, it never separately runs `pip install -e .` the way local
  setup did, so the project's own package was never actually installed. Fixed
  by adding a trailing `-e .` line to `requirements.txt` itself.
- **`confluent-kafka` (transitive, via Hopsworks) failed to build from
  source** — `fatal error: librdkafka/rdkafka.h: No such file or directory`.
  No prebuilt wheel for this platform/Python combo, and the system library
  header it needs isn't in Streamlit Cloud's base image. Fixed with a new
  `packages.txt` (Streamlit Cloud's mechanism for apt-installable system
  deps) listing `librdkafka-dev`.

Secrets **are** required (this note previously said otherwise, written before
Phase 4.5's real Hopsworks integration existed) — `HOPSWORKS_API_KEY` and
`HOPSWORKS_PROJECT_NAME`, added via the app's Secrets panel in TOML format,
same two values used everywhere else in this project.

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

**Verified end to end against project `p/43151` (eu-west serverless):**
- `migrate_to_hopsworks.py`: 18,336 rows → `aqi_features` v1, materialization job
  SUCCEEDED, read-back 18,336 / 52 cols, count matches the local parquet.
- all four `tests/smoke_*.py` pass, no network.
- `train.py`: `us_aqi_next` **v1** registered (random_forest, test RMSE 0.155).
- `train_multi_horizon.py`: `us_aqi_h24` / `h48` / `h72` **v1** registered
  (all XGBoost; test RMSE 4.949 / 7.910 / 10.371, R² 0.818 / 0.534 / 0.199 —
  identical to Phase 2c, confirming the migrated data matches).
- `predict.py --location karachi`: live 4-horizon forecast from the newly
  registered Hopsworks models.
- Hopsworks UI shows the `aqi_features` feature group and all 4 model names.

**Dependency fallout (`requirements.txt`).** `hopsworks[python]==5.0.6` pins
`pandas<2.4`, `numpy<2.5`, `protobuf<5`. To make one venv resolve:
- `tensorflow==2.21.0` **dropped** — nothing imports it, and its
  `protobuf>=6.31.1` pin is irreconcilable with Hopsworks. (`torch` stays; it is
  still only used by the deferred, unbuildable `lstm_model.py`.)
- `pandas` 3.0.5 → **2.3.3**, `streamlit` 1.62.0 → **1.59.1**.
- `+ deltalake==1.6.3` — hopsworks needs a delta-rs lib for non-Spark ops and
  ships `hops-deltalake` for it, but that wheel is linux / macos-arm only, so
  upstream `deltalake` is used. (In the end the feature group is HUDI, not
  DELTA — see the `store.py` note above — but hopsworks imports the module
  regardless, and it is the natural path if the offline store is ever S3.)
- `twofish` (transitive via `pyjks` via `hopsworks`) ships no wheel and there is
  no C compiler on this machine; it is satisfied by a pure-Python import shim
  (`twofish` 0.3.0, built from `scripts/`-adjacent throwaway — not vendored) that
  raises if the cipher is ever actually exercised. The API-key REST login path
  never touches it. On a machine with MSVC build tools,
  `pip install --force-reinstall --no-binary twofish twofish` swaps in the real
  one. **`requirements.txt` still lists `twofish` only implicitly** (it is a
  transitive dep); the shim is installed separately and is not in the file.

**Windows / Serverless notes.** Getting an external `hopsworks[python]` client to
write a feature group from Windows took three fixes, all in `hopsworks_client.py`
or `store.py`: (1) `cert_folder` override — the `/tmp` default can't be created
on Windows; (2) pre-create `<drive>:\tmp` — the Kafka connector's PEM export
hardcodes `/tmp`; (3) HUDI + `statistics_config=False` — the DELTA/delta-rs path
needs a direct HDFS namenode connection the serverless project doesn't expose
externally, and the HUDI Spark job's built-in statistics step was returning a
backend 500. With those, `migrate_to_hopsworks.py` runs clean: 18,336 rows, the
materialization job SUCCEEDS, and the read-back count matches the local parquet.

Out of scope: GitHub Actions/CI, Streamlit Cloud deploy, hazardous-AQI alerting,
LSTM.

## Phase 5 – Dashboard redesign, SHAP explainability & GitHub Actions automation — done

*(This section replaces the earlier "not started" stub, which was never updated
after the work below actually landed across commits `70065e6`, `6052e42`,
`bf9a533`, `2536ac1`, `7ffaaef`.)*

### Dashboard redesign (`70065e6`)

`aqi_predictor/dashboard/app.py` moved from a plain title + two line charts to
a layout closer to real public AQI apps:

- four `st.columns(4)` stat cards (Now / +24h / +48h / +72h), custom CSS
  (`_CARD_CSS`), each showing the AQI number and a category pill coloured from
  `aqi_scale.aqi_category()`, with `_text_on()` picking readable text colour
  against each pill's background by luminance.
- `_trend_chart()` — an Altair *layered* chart: `mark_rect` EPA category bands
  behind a `mark_line` of observed AQI. Both layers need explicit `x`/`x2`
  spans across the observed window; the "omit x for a full-width band" idiom
  does not reliably render in Vega-Lite.
- `_forecast_chart()` — a *bar* chart, not a connected line: four discrete
  horizon predictions aren't a continuous series, and drawing them as one
  would visually imply interpolation that isn't real. Bars coloured per-point
  by category, values labelled via `mark_text`.
- raw forecast table moved into a collapsed `st.expander("Forecast details")`.

Two real bugs only surfaced by actually running the app, not from reading the
code: Hopsworks/hsfs's `tqdm` progress bars were stalling Streamlit's render
entirely (`os.environ.setdefault("TQDM_DISABLE", "1")` fixes it), and Vega-Lite
was silently dropping every row when fed tz-aware ISO timestamps (fixed by
stripping tz before charting, same pattern `store.py` already used).

### SHAP explainability (`6052e42`)

Split by design into two separate computation points, not one:

- **Global** (training time) — `train.compute_shap_importance(model, X_test)`
  runs `shap.TreeExplainer(model).shap_values(X_test)` and returns mean
  `|SHAP|` per feature, only for `RandomForestRegressor`/`XGBRegressor`
  (`isinstance` check); `None` for anything else (Ridge is wrapped in a
  `Pipeline` and structurally can't match). Stored in the registered model's
  `metadata.json` as `shap_importance` — additive, doesn't change
  `register_model`'s existing callers.
- **Local** (inference time) — `predict._local_shap_top_features(model, x)`
  runs the same `TreeExplainer` on the *exact* single input row `x` that was
  actually fed to `model.predict()` for that forecast, ranked by `|SHAP|`, top
  5 kept. This is per-prediction, not per-model.
- Dashboard: a new "Explain this forecast" section — horizon selector, a local
  SHAP bar chart coloured by push direction (positive = pushes AQI up), and a
  global mean-`|SHAP|` bar chart, both degrading to a plain caption when the
  served model has no SHAP data (Ridge).
- `shap.TreeExplainer` was the deliberate choice over `KernelExplainer` (the
  slow, no-background-needed alternative used by the earlier, now-deleted LSTM
  code): tree explainers need no background sample and are exact, not
  approximate, for tree models.
- `requirements.txt`: `+ shap==0.52.0`.

### Registry tie-break fix (surfaced by Claude Code's own multi-option prompts, decided live)

Two real, separate bugs in `load_best_model`'s "lowest test RMSE wins" rule:

1. An **exact tie** between the old pre-SHAP v1 and new v2 for `us_aqi_next`
   and `us_aqi_h72` (deterministic retraining on unchanged data) meant the
   dashboard non-deterministically served v1 depending on Hopsworks' API
   return order. Fixed by a deliberately general rule, not a SHAP-specific
   hack: **prefer the higher version number on any RMSE tie.**
2. A **near-tie** (~1.7e-16 apart — RandomForest's `n_jobs=-1` parallel-
   reduction floating-point noise) that the exact-equality fix above didn't
   catch. Fixed by rounding test RMSE to `_RMSE_COMPARISON_PRECISION = 6`
   decimal places before comparing, in both the fast path (Hopsworks' flat
   `training_metrics`) and the fallback path (downloaded `metadata.json`), sort
   key `(round(rmse, 6), -version)`. Verified against the exact real numbers
   hit in production: `0.15454074049236596` vs `0.15454074049236613`.

### GitHub Actions automation (`70065e6`, workflow files)

- `.github/workflows/feature_pipeline.yml` — hourly (`cron: "17 * * * *"`, off
  the top-of-hour rush) + `workflow_dispatch`, `concurrency: {group:
  feature-pipeline, cancel-in-progress: false}` so overlapping runs queue
  instead of racing. Runs the new `scripts/run_feature_pipeline.py`.
- `.github/workflows/training_pipeline.yml` — daily (`cron: "40 2 * * *"`
  UTC) + `workflow_dispatch`. Runs `train.py` → `check_registry_after_training.py
  --name us_aqi_next` → `train_multi_horizon.py` → the same check for
  `us_aqi_h24`/`h48`/`h72`.
- Both set `HOPSWORKS_API_KEY`/`HOPSWORKS_PROJECT_NAME` from
  `${{ secrets.* }}` as job-level env vars.
- `scripts/run_feature_pipeline.py` deliberately re-fetches a rolling
  `DEFAULT_HISTORY_DAYS = 4` window via `fetch.historical()` +
  `build_features()` — **not** `fetch.latest_hour()` — because lag/rolling
  features need days of prior context a single new row can't supply. This
  turned out to also make the pipeline self-healing against GitHub's
  scheduling unreliability (below): a skipped hour's data is picked up by the
  next run automatically.
- `scripts/check_registry_after_training.py` compares `list_versions(name)`'s
  newest version against what `load_best_model(name)` actually serves, and
  logs which one won. Deliberately no promotion-gating: `load_best_model`
  already always serves the global-best-ever version, so a bad daily retrain
  is harmless to what's served — logging the outcome is enough.

### CI/CD debugging chain — four independent root causes, each diagnosed from a real error log, not guessed

1. **`ERROR: No matching distribution found for xgboost==3.3.0`** on Python
   3.11 runners — xgboost 3.3.0 requires Python ≥3.12 (confirmed against
   PyPI's release history), while the real local `.venv` runs 3.13.3. Fixed:
   `python-version` `"3.11"` → `"3.13"` in both workflow files.
2. **`HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT_NAME` are not set`** — the
   GitHub repo secrets had never actually been added. Fixed by adding both
   manually under Settings → Secrets and variables → Actions.
3. **`pyarrow._flight.FlightUnavailableError: Socket closed`** reading the
   feature group — Hopsworks' Arrow Flight Query Service (a separate
   port/service from the already-authenticated HTTPS API) is unreachable from
   GitHub-hosted runners' rotating IPs. Fixed by adding
   `read_options={"use_hive": True}` to `store.py`'s `.read()` call, forcing
   the documented Hive/Spark fallback instead of Arrow Flight. Confirmed
   working after deploy (commit `7ffaaef`).
4. **The hourly feature-pipeline cron actually firing every 2–4 hours, not
   hourly** — confirmed as a known, documented GitHub Actions limitation
   (`schedule` is best-effort shared infrastructure, not a guarantee), not a
   config bug. Accepted as-is: the 4-day rolling fetch window in
   `run_feature_pipeline.py` makes a delayed or skipped run self-healing, with
   no real data-completeness cost.

Out of scope for this phase: the LSTM (Phase 6), statistical baselines
(Phase 7), EDA/alerting (Phase 8), and Streamlit Cloud deployment (still
manual, see Phase 4).

## Phase 6 – LSTM deep learning model — done

**What changed.** `training_pipeline/lstm_model.py` (previously stale –
imported a deleted `process.py`, used `KernelExplainer`, was a top-level script
that ran training on import rather than an importable module) was rewritten
from scratch:

- `AQI_LSTM(nn.Module)` – 1-2 layer LSTM (default: 1 layer, hidden size 64,
  dropout 0.2) + a linear head. `seq_len` is stored as a plain attribute so it
  survives `joblib` pickling and tells `predict.py` how much history to fetch.
  Forward returns `(batch, 1)`, not squeezed to `(batch,)` – `shap`'s PyTorch
  wrapper indexes `outputs.shape[1]` and errors on a 1-D output.
- `build_sequences(df, feature_columns, target, seq_len=48)` – slides a
  48-hour window per location over **one split at a time** (train/val/test),
  so windows never cross a split boundary; a window is dropped if any pair of
  its 48 hours is not exactly 1h apart (an unfilled gap), so a gap never
  silently splices non-adjacent hours together.
- `train_lstm(splits)` – fits a **train-only** `StandardScaler` (the tree
  models need no scaling; this one does), trains with Adam + early stopping on
  validation RMSE (patience 8, max 100 epochs), and returns the model, metrics,
  a `{"scaler.joblib": ..., "shap_background.joblib": ...}` dict for the
  registry, and an `lstm_context` dict (scaled test sequences + SHAP
  background) so `train.py` doesn't have to re-derive them for the global SHAP
  computation.

**Sequence vs. tabular reasoning.** The tree/linear models see one engineered
feature row per prediction; the LSTM sees the trailing 48-hour trajectory of
the *same* engineered columns (not raw un-engineered inputs – lags, rolling
stats, the `<var>_target` future-weather features, etc. are all reused as-is).
Same per-horizon `us_aqi` target, same train/val/test split boundaries, no
shuffling. It registers under the same four model names
(`us_aqi_next`/`us_aqi_h24`/`h48`/`h72`) and competes for real via
`registry.load_best_model`'s existing lowest-test-RMSE tie-break – there is no
special-casing to make it win or lose.

**Registry (`registry.py`) – generalized for extra artifacts.**
`register_model(..., extra_artifacts: dict[str, Any] | None = None)` joblib-
dumps each `{filename: object}` pair alongside `model.joblib`; `metadata.json`
records only the filenames (`extra_artifact_files`). `load_model`/
`load_best_model` keep their existing 2-tuple return shape – no signature
change – but load every named extra artifact back and attach them as
`metadata["extra_artifacts"] = {filename: object}`. Models registered without
`extra_artifacts` (the four tree/linear models, plus every version registered
before this existed) get back `{}`, confirmed against real already-registered
versions in Hopsworks (`us_aqi_next` v1-v4 have no `extra_artifact_files` key
at all – `.get(...) or []` handles the missing key cleanly).

**Inference (`predict.py`) – isinstance dispatch.** `forecast()` now branches
on `isinstance(model, AQI_LSTM)`: the sequence path pulls the trailing
`model.seq_len` hours from `observed`, builds per-row `<var>_target` values via
a new `build_input_sequence()` (each window row's own `time + horizon`, not
just the last row's – `forecast_ahead`'s `past_days=2` window comfortably
covers the full span needed at every horizon), scales with the stored scaler,
and runs the model in eval mode under `torch.no_grad()`. The existing
single-row tree-model path (`build_input_row`, `_local_shap_top_features`) is
untouched.

**SHAP: DeepExplainer → GradientExplainer (a deviation from the original
plan, found in-session).** `shap.DeepExplainer` was tried first as specified.
It has no attribution rule registered for `nn.LSTM`
(`UserWarning: unrecognized nn.Module: LSTM`), and the resulting attributions
failed shap's own additivity check by ~40x tolerance (`Max. diff:
0.399 - Tolerance: 0.01`) against the real trained model – not a rounding
error, a genuinely unsupported op. Switched to `shap.GradientExplainer`
(same package, still gradient-based/fast, no per-layer op registry so it works
on any differentiable architecture) after checking with the user rather than
silently suppressing the additivity check or dropping SHAP for the LSTM.
Global importance (`train.py`) sums `|SHAP|` across the time axis then
averages over samples, matching the tree path's existing non-negative
`{feature: mean_abs_shap}` contract. Local importance (`predict.py`) sums the
*signed* SHAP values across the time axis instead – preserving the direction
the tree path already carries (positive = pushes AQI up) so the dashboard's
existing up/down coloring on the local SHAP chart stays meaningful; this is a
deliberate deviation from a literal "always take the absolute value" reading,
made to keep both existing dashboard contracts (signed local / non-negative
global) intact without any dashboard code change.

**Real training results (Karachi, full Hopsworks feature store, test split;
2026-09-06).** LSTM competed honestly against Ridge/RandomForest/XGBoost –
no thumb on the scale either way:

| horizon | ridge  | random_forest | xgboost | **lstm**  | winner (test RMSE) |
|---------|--------|----------------|---------|-----------|---------------------|
| +1h     | 0.624  | 0.285          | 0.283   | 0.590     | xgboost (v5) – but a pre-existing random_forest v4 (RMSE 0.1545, from before this phase) still beats all of v5 and is what `load_best_model` actually serves |
| +24h    | 5.778  | 5.326          | 4.887   | **4.810** | **lstm** |
| +48h    | 10.109 | 8.353          | 8.014   | **7.343** | **lstm** |
| +72h    | 12.288 | 11.645         | 11.012  | **9.944** | **lstm** |

The LSTM lost at +1h (where persistence dominates and the tree models were
already near-perfect) and won at every multi-step horizon, where the 48h
trajectory gives it signal the single-row tree models don't have. Both
outcomes are reported as-is.

**Registered in Hopsworks (definition-of-done runs).**
`python -m aqi_predictor.training_pipeline.train` → `us_aqi_next` v5
(xgboost; served version is still v4, random_forest, per the RMSE table
above). `python -m aqi_predictor.training_pipeline.train_multi_horizon` →
`us_aqi_h24` v5, `us_aqi_h48` v5, `us_aqi_h72` v4 (lstm, all three) – each
upload included `scaler.joblib` + `shap_background.joblib` via
`extra_artifacts`; Hopsworks' model registry accepted the extra-artifact
upload approach with no issue (the "stop and tell me if the registry rejects
this" condition did not trigger).
`python -m aqi_predictor.inference_pipeline.predict --location karachi`
produced a forecast at all four horizons against this real mixed fleet
(random_forest serving +1h, lstm serving +24h/+48h/+72h) with no crash.

**Tests.** `tests/smoke_training_pipeline.py` – `build_sequences` shape/no-
leakage/gap-handling/no-cross-location-window checks, an end-to-end tiny
(5-epoch) `train_lstm` run on synthetic data through
`shap.GradientExplainer` and the registry's `extra_artifacts` round-trip.
`tests/smoke_inference_pipeline.py` – `build_input_sequence`'s per-row
target-time lookup and its missing-feature error, `_lstm_local_shap`'s ranked
output shape and `None`-on-no-background behavior. `tests/smoke_dashboard.py`
– the "Explain this forecast" section renders for a forecast whose
`top_features`/`shap_importance` are shaped like the LSTM's output, confirming
no dashboard code assumed a tree-model-only shape (none needed changing).
All four `tests/smoke_*.py` pass, no network, no full-size training.
`python -m pytest tests/ -v` collects 0 items and "passes" trivially – this
project's tests are plain scripts (`python tests/smoke_*.py`), not
pytest-style (`test_*` files/functions), and pytest itself isn't in
`requirements.txt`; the real verification is the four smoke-test runs above.

Out of scope for this phase (untouched, as instructed): how the tree/linear
models are trained, registered, or served; registry's other public function
signatures; hazardous-AQI alerting.

## Phase 7 – Statistical baselines — done

**What changed.** `scripts/evaluate_baselines.py` computes two classical
baselines and combines them with the already-registered ML models' test
metrics into one comparison table, saved to `reports/model_comparison.csv`.
Neither baseline calls `registry.register_model` and neither is ever eligible
to be served – they exist purely for this table. `load_best_model`'s
selection logic is untouched; the script only *calls* it (to read metrics),
never modifies it.

- **seasonal-naive** – prediction for any target time = the actual observed
  `us_aqi` exactly 24h before that target time, regardless of horizon (so for
  +24h this is literally "today's value predicts tomorrow's"; for +1h/+48h/
  +72h it's the same fixed daily-period rule applied uniformly, per spec).
- **SARIMA** – `statsmodels` `SARIMAX`, one fit per location on that
  location's pre-test history only (no test leakage), univariate on `us_aqi`,
  fixed order `(1,1,1)x(1,1,1,24)` (not grid-searched – a comparison
  baseline). Added `statsmodels==0.15.0` to `requirements.txt` (pmdarima was
  explicitly not added, per the brief).
- Both baselines are evaluated against the *same* `dataset.split_dataset`
  test-split boundary the ML models used – the script fetches the feature
  store once and passes it into `build_training_frame(df=..., horizon_hours=h)`
  per horizon, replicating exactly what `train.py`/`train_multi_horizon.py`
  did rather than approximating it.
- The ML rows are pulled from `registry.load_best_model(name)` per horizon
  (whatever is actually being served right now), not retrained, labeled
  `location="all"` since those models are trained pooled across every
  location in the feature store, not fit per location like the baselines.

**Real results (Karachi, full Hopsworks feature store, test split;
2026-09-06)** – see `reports/model_comparison.csv` for the full table:

| horizon | seasonal_naive | sarima | **served ML** |
|---------|----------------|--------|----------------|
| +1h     | RMSE 7.245     | RMSE 27.197 | RMSE 0.155 (random_forest) |
| +24h    | RMSE 7.261     | RMSE 26.883 | RMSE 4.810 (lstm) |
| +48h    | RMSE 7.368     | RMSE 26.883 | RMSE 7.343 (lstm) |
| +72h    | RMSE 7.369     | RMSE 26.668 | RMSE 9.944 (lstm) |

The registered ML models beat both baselines at every horizon. **SARIMA
underperforms even the naive baseline, badly** (R2 around -4.2 to -4.6, i.e.
worse than predicting the mean) – checked this wasn't an indexing/timezone
bug before accepting it: a manual trace showed the fitted model tracks the
actual series closely for the first ~60 forecast steps, then collapses to a
near-flat ~71 for the remaining ~350 of the 409 static forecast steps
required, losing the daily oscillation (actual data swings 60-92) entirely.
That is a real property of a fixed, non-grid-searched `(1,1,1)x(1,1,1,24)`
order asked to produce one static ~17-day-ahead forecast with no
re-fitting – not a bug in the evaluation code. Reported as-is, per the same
"legitimate result, not a failure to hide" principle Phase 6 used for the
LSTM.

**Tests.** `tests/smoke_evaluate_baselines.py` – seasonal-naive lookup
correctness (any horizon, any missing-lookup case returns NaN rather than
crashing), `_metric_row` excluding NaN predictions from RMSE/MAE/R2, and a
SARIMA fit + forecast on a 200-hour *synthetic* series (no network, runs in
under a second) checking the forecast is finite and correctly time-indexed
past training data. All five `tests/smoke_*.py` files pass, no network calls
added to the test suite.

Out of scope for this phase (as instructed): calling `register_model` for
either baseline; changing `load_best_model`'s selection logic; tuning SARIMA
beyond the fixed order.

## Phase 8 – EDA + hazardous-AQI alerting — done

**What changed.**

- **`aqi_predictor/dashboard/eda_charts.py`** (new) – four chart functions
  (`trend_and_gaps`, `seasonality_patterns`, `category_distribution`,
  `feature_correlation`), each taking a single-location engineered feature
  frame and returning an Altair chart. Shared verbatim by the notebook and the
  new dashboard page – neither writes the analysis logic itself.
  `alt.data_transformers.disable_max_rows()` is set at module import: Altair's
  default 5000-row cap doesn't survive contact with an 18k+ row real feature
  store (`MaxRowsError`, hit while building the notebook – see below).
  `trend_and_gaps` additionally resamples its *line* to a daily mean above
  ~2000 hourly points (gap detection still runs on the full hourly index, so
  a gap doesn't need to span a whole day to still show), keeping the embedded
  Vega-Lite spec small.
- **`notebooks/eda.ipynb`** (new) – four short sections (one chart + a
  couple of sentences each) built with `nbformat`/`nbclient` against the real
  Hopsworks feature store + model registry, executed so outputs are saved in
  the file. `nbformat`/`nbclient`/`ipykernel` were installed locally to author
  and run it once; they are not added to `requirements.txt` since nothing in
  the shipped app/pipelines needs them at runtime.
- **`aqi_predictor/dashboard/pages/1_Data_Insights.py`** (new) – Streamlit's
  native multipage convention (a `pages/` folder next to `app.py`, which had
  no existing `pages/` to conflict with). Same location-selector pattern as
  `app.py`'s `main()` (`st.selectbox` over `LOCATIONS`, `format_func=str.title`)
  – no second way to pick a location. `store.get_feature_view(...)` is wrapped
  in `st.cache_data(ttl=1200)` so navigating back and forth doesn't re-hit
  Hopsworks.
- **`aqi_scale.py`** – added `CATEGORIES` (the full ordered `(label, colour)`
  list, derived from the existing private breakpoint data instead of a second
  hardcoded copy) and `hazardous_horizons(current, forecasts)`, which checks
  `predict.forecast()`'s own `current`/`forecasts` dicts against the real
  "Hazardous" threshold already in `aqi_category` (> 300) and returns which
  entries triggered it (e.g. `["now", "+24h"]`).
- **`app.py`** – calls `hazardous_horizons` right after `load_forecast`
  succeeds (no new fetch) and renders a red, top-of-page banner above the
  stat cards naming the triggering horizon(s) when the list is non-empty; no
  banner otherwise. Banner-only: no email, no new GitHub secret, no workflow
  change, no dedup/cooldown – it's a live check computed at render time, not
  a pipeline step.

**Real seasonality findings (Karachi, full Hopsworks feature store;
2026-09-06)** – the user asked whether this is actually interesting before
it goes in the final report:

- **Month-of-year: yes, strongly.** Mean `us_aqi` swings ~36 points across
  the year – highs of 108.8 in December/January, a low of 72.9 in September.
  Winter (Nov-Feb, 94-109) vs. monsoon season (Aug-Sep, 73-75) is a large,
  physically sensible gap (temperature inversions + winter burning vs.
  monsoon rain washing out particulates) and is worth featuring.
- **Hour-of-day: weak, in UTC.** Only a ~5.9-point range - flat at ~86.4-86.5
  most of the day with a modest bump to 92.3 around 13:00-14:00 UTC
  (18:00-19:00 Karachi local time, i.e. evening rush hour/cooking - plausible,
  but a small effect in the aggregate). Worth a passing mention, not a
  headline.
- **Correlation vs. SHAP cross-check (us_aqi_next, served v4 random_forest):**
  raw correlation with `us_aqi` ranks `pm2_5` highest (0.726); the served
  model's SHAP global importance (excluding `us_aqi` itself, trivially
  dominant for a 1h-ahead near-persistence task) ranks `pm2_5_roll_mean_24h`
  highest. Both independently point to pm2.5 as the dominant driver - a
  useful, honest cross-check rather than a discrepancy.

**Definition-of-done verification.** The Claude-in-Chrome browser extension
was not connected in this environment, so a literal click-through visual
check wasn't possible - said so rather than claiming it. Verified instead
with the strongest available non-browser equivalents: `streamlit run` in the
background + `curl` confirmed both `/` and `/Data_Insights` return HTTP 200
(Streamlit's own multipage routing picked up the new page); real
(un-mocked) `AppTest` runs against live Hopsworks data for both scripts
showed no exceptions, all four `Data Insights` section headers rendering,
and the hazard banner correctly absent under real current conditions (no
horizon is currently Hazardous). The forced-hazardous case is covered by
`tests/smoke_dashboard.py`'s `check_hazard_banner`, which runs the real
`app.py` end to end via `AppTest` with a forced Hazardous value and asserts
the banner text appears (and a separate clean run asserts it's absent) -
the same mechanism used throughout this project's dashboard tests, just
short of an actual browser window.

**Tests.** `tests/smoke_eda_charts.py` (new) – each of the four chart
functions runs against a small synthetic dataframe with no crash, plus a
large-synthetic-dataset check exercising the daily-resample branch.
`tests/smoke_dashboard.py` – added `check_hazard_banner` (forces "now" into
Hazardous, asserts the banner text and triggering horizon appear; a clean
forecast asserts it's entirely absent - the first assertion attempt
false-positived on the `.hazard-banner` CSS class always being present in
injected `<style>`, fixed by checking for the banner's own text instead of
the class name). All six `tests/smoke_*.py` files pass.

Out of scope for this phase (as instructed): the training/registry/predict
pipelines; any GitHub Actions secret or workflow change; a second
location-selection UI.
