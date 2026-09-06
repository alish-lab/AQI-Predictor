"""Statistical baselines (seasonal-naive, SARIMA) vs. the registered ML models.

Computes seasonal-naive and SARIMA forecasts + RMSE/MAE/R2 per location and
horizon, over the exact same test-split boundary the ML models
(``training_pipeline/train.py`` / ``train_multi_horizon.py``) were evaluated
on, then combines them with those models' already-registered test metrics
(pulled from the registry, not retrained) into one comparison table.

Neither baseline is registered via ``registry.register_model``: they exist
purely for this comparison table, are never eligible to be served, and this
script does not touch ``load_best_model``'s selection logic at all.

* seasonal-naive - prediction for any target time = the actual observed
  ``us_aqi`` exactly 24h before that target time (standard seasonal-naive,
  daily period), regardless of forecast horizon.
* SARIMA - ``statsmodels`` ``SARIMAX``, one fit per location on that
  location's pre-test history, univariate on ``us_aqi``, fixed order
  ``(1,1,1)x(1,1,1,24)`` (a comparison baseline, not grid-searched).

Run::

    python scripts/evaluate_baselines.py
    python -m scripts.evaluate_baselines
"""

from __future__ import annotations

import csv
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from aqi_predictor.config import PROJECT_ROOT
from aqi_predictor.feature_pipeline import store
from aqi_predictor.inference_pipeline.predict import HORIZONS, model_name
from aqi_predictor.training_pipeline import dataset, registry
from aqi_predictor.training_pipeline.metrics import regression_metrics

REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
REPORT_COLUMNS = ["model", "location", "horizon", "rmse", "mae", "r2"]

# Mirrors dataset.py's private _FULL_START/_FULL_END: fetching the feature
# store once here and passing it into build_training_frame(df=...) for every
# horizon avoids a redundant Hopsworks round trip per horizon.
_FULL_START = "2000-01-01"
_FULL_END = "2100-01-01"

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)


def _seasonal_naive_predictions(
    test_df: pd.DataFrame, raw_df: pd.DataFrame, horizon_hours: int
) -> np.ndarray:
    """Prediction for each test row = the actual observed ``us_aqi`` exactly
    24h before that row's target time, looked up from ``raw_df`` (the
    undropped feature-store history, so lookups near the split boundary still
    resolve)."""
    lookup_time = test_df["time"] + pd.Timedelta(hours=horizon_hours - 24)
    key = pd.DataFrame(
        {"location": test_df["location"].to_numpy(), "time": lookup_time.to_numpy()}
    )
    merged = key.merge(
        raw_df[["location", "time", "us_aqi"]], on=["location", "time"], how="left"
    )
    return merged["us_aqi"].to_numpy()


def _fit_sarima(train_series: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SARIMAX is chatty about convergence/freq
        model = SARIMAX(
            train_series,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def _metric_row(model: str, location: str, horizon: int, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_pred)
    dropped = len(y_pred) - int(mask.sum())
    if dropped:
        print(
            f"[baselines] {model} {location} +{horizon}h: {dropped}/{len(y_pred)} "
            f"row(s) had no prediction available, excluded from metrics",
            flush=True,
        )
    m = regression_metrics(y_true[mask], y_pred[mask])
    return {"model": model, "location": location, "horizon": horizon, **m}


def evaluate_baselines() -> list[dict]:
    """Seasonal-naive + SARIMA metrics per location/horizon, over the exact
    same test split ``dataset.split_dataset`` gives the ML models."""
    raw_df = store.get_feature_view(_FULL_START, _FULL_END)
    if raw_df.empty:
        raise RuntimeError("feature store returned no rows; run scripts/backfill.py first")

    splits_by_horizon = {
        h: dataset.split_dataset(
            dataset.build_training_frame(raw_df, horizon_hours=h), horizon_hours=h
        )
        for h in HORIZONS
    }

    locations = sorted(raw_df["location"].unique())
    rows: list[dict] = []

    for location in locations:
        loc_tests = {
            h: splits_by_horizon[h].test.loc[
                splits_by_horizon[h].test["location"] == location
            ]
            for h in HORIZONS
        }
        loc_tests = {h: df for h, df in loc_tests.items() if not df.empty}
        if not loc_tests:
            continue

        test_start = min(df["time"].min() for df in loc_tests.values())
        max_target_time = max(
            df["time"].max() + pd.Timedelta(hours=h) for h, df in loc_tests.items()
        )

        hist = (
            raw_df[(raw_df["location"] == location) & (raw_df["time"] < test_start)]
            .sort_values("time")
            .set_index("time")["us_aqi"]
            .asfreq("h")
        )
        steps = int((max_target_time - hist.index.max()) / pd.Timedelta(hours=1))
        print(
            f"[baselines] fitting SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} for "
            f"{location} on {len(hist)}h of history, forecasting {steps} steps ahead ...",
            flush=True,
        )
        fitted = _fit_sarima(hist)
        sarima_forecast = fitted.get_forecast(steps=steps).predicted_mean

        for horizon, loc_test in loc_tests.items():
            target = splits_by_horizon[horizon].target
            y_true = loc_test[target].to_numpy()

            naive_pred = _seasonal_naive_predictions(loc_test, raw_df, horizon)
            rows.append(_metric_row("seasonal_naive", location, horizon, y_true, naive_pred))

            target_times = loc_test["time"] + pd.Timedelta(hours=horizon)
            sarima_pred = sarima_forecast.reindex(target_times).to_numpy()
            rows.append(_metric_row("sarima", location, horizon, y_true, sarima_pred))

    return rows


def ml_registered_rows() -> list[dict]:
    """Test metrics for the currently-served (lowest-test-RMSE) model at each
    horizon, pulled from the registry - never retrained, never re-registered.

    Delegates the actual ``load_best_model`` loop to
    ``registry.current_model_metrics`` (also used by the dashboard's "Model
    Performance" table) rather than re-implementing it here.
    """
    names = [model_name(h) for h in HORIZONS]
    by_name = {m["name"]: m for m in registry.current_model_metrics(names)}

    rows: list[dict] = []
    for horizon in HORIZONS:
        name = model_name(horizon)
        m = by_name.get(name)
        if m is None:
            print(f"[baselines] no registered model named {name!r}, skipping", flush=True)
            continue

        rows.append(
            {
                "model": f"{m['algorithm']} (served v{m['version']})",
                # the registered models are trained pooled across every location in
                # the feature store, not fit per location like the baselines -
                # "all" is honest about that rather than implying a per-location fit.
                "location": "all",
                "horizon": horizon,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
        )
    return rows


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def print_table(rows: list[dict]) -> None:
    header = f"{'model':<28}{'location':<12}{'horizon':>9}{'RMSE':>10}{'MAE':>10}{'R2':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<28}{r['location']:<12}{'+' + str(r['horizon']) + 'h':>9}"
            f"{_fmt(r['rmse'], '>10.3f')}{_fmt(r['mae'], '>10.3f')}{_fmt(r['r2'], '>10.4f')}"
        )


def write_report(rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in REPORT_COLUMNS})


def main() -> int:
    rows = evaluate_baselines()
    rows.extend(ml_registered_rows())
    rows.sort(key=lambda r: (r["horizon"], r["location"], r["model"]))

    print("\n" + "=" * 79)
    print("MODEL COMPARISON - statistical baselines vs. registered ML models")
    print("=" * 79)
    print_table(rows)
    print("=" * 79)

    write_report(rows)
    print(f"\nsaved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
