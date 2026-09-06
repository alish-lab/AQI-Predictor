"""Offline smoke checks for scripts/evaluate_baselines.py (no network).

Run:  python tests/smoke_evaluate_baselines.py

Covers:
* seasonal-naive lookup - predicts the actual value exactly 24h before the
  target time, for any horizon, and returns NaN (not a crash) when that
  24h-earlier observation isn't available.
* SARIMA fit + forecast on a tiny (200h) synthetic series - no network, no
  real Hopsworks data - produces a finite, correctly time-indexed forecast.
* ``_metric_row`` drops NaN predictions from the metric computation rather
  than letting them poison RMSE/MAE/R2.
* ``ml_registered_rows`` still produces the same row shape/values now that it
  delegates its ``load_best_model`` loop to
  ``registry.current_model_metrics`` - checked against an in-memory fake
  Hopsworks registry (no network), the same pattern
  ``smoke_training_pipeline.py`` uses.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# scripts/ isn't an installed package (only aqi_predictor is, via `pip install
# -e .`), so make the repo root importable regardless of this test's own cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import scripts.evaluate_baselines as evaluate_baselines
from aqi_predictor.training_pipeline import registry
from scripts.evaluate_baselines import (
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    _fit_sarima,
    _metric_row,
    _seasonal_naive_predictions,
)


# --------------------------------------------------------------------------- #
# In-memory stand-in for the Hopsworks Model Registry (no network) - same
# minimal pattern as smoke_training_pipeline.py's fakes, kept self-contained
# here rather than cross-imported so this file still runs standalone.
# --------------------------------------------------------------------------- #
class _FakeModel:
    def __init__(self, mr: "_FakeModelRegistry", name: str, metrics: dict) -> None:
        self._mr = mr
        self.name = name
        self.training_metrics = dict(metrics)
        self.version: int | None = None
        self._dir: Path | None = None

    def save(self, src_dir: str) -> "_FakeModel":
        self.version = self._mr._next_version(self.name)
        dst = Path(tempfile.mkdtemp(prefix=f"fakemr_{self.name}_v{self.version}_"))
        for item in Path(src_dir).iterdir():
            shutil.copy2(item, dst / item.name)
        self._dir = dst
        self._mr._store[(self.name, self.version)] = self
        return self

    def download(self) -> str:
        return str(self._dir)


class _FakeModelRegistryNS:
    def __init__(self, mr: "_FakeModelRegistry") -> None:
        self._mr = mr

    def create_model(self, name: str, metrics: dict, description: str = "") -> _FakeModel:
        return _FakeModel(self._mr, name, metrics)


class _FakeModelRegistry:
    def __init__(self) -> None:
        self._store: dict[tuple[str, int], _FakeModel] = {}
        self.python = _FakeModelRegistryNS(self)

    def _next_version(self, name: str) -> int:
        existing = [v for (n, v) in self._store if n == name]
        return (max(existing) + 1) if existing else 1

    def get_models(self, name: str) -> list[_FakeModel]:
        return [m for (n, _v), m in self._store.items() if n == name]

    def get_model(self, name: str, version: int) -> _FakeModel | None:
        return self._store.get((name, version))


class _FakeProject:
    def __init__(self, mr: _FakeModelRegistry) -> None:
        self._mr = mr

    def get_model_registry(self) -> _FakeModelRegistry:
        return self._mr


def _synthetic_raw(n_hours: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    seasonal = 70 + 15 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    values = seasonal + rng.normal(0, 2, n_hours)
    return pd.DataFrame({"location": "alpha", "time": idx, "us_aqi": values})


def check_seasonal_naive_predictions() -> None:
    raw = _synthetic_raw(60)
    test_df = raw.iloc[-10:].reset_index(drop=True)

    for horizon in (1, 24, 48):
        pred = _seasonal_naive_predictions(test_df, raw, horizon)
        lookup_time = test_df["time"] + pd.Timedelta(hours=horizon - 24)
        expected = raw.set_index("time")["us_aqi"].reindex(lookup_time).to_numpy()
        np.testing.assert_allclose(pred, expected, equal_nan=True)

    print(
        "ok  seasonal_naive: predicts the value exactly 24h before the "
        "target time, for any horizon"
    )


def check_seasonal_naive_missing_lookup_is_nan() -> None:
    raw = _synthetic_raw(10)
    test_df = raw.iloc[[0]].reset_index(drop=True)
    # target time (row + 1000h) is far outside raw's covered range -> NaN
    pred = _seasonal_naive_predictions(test_df, raw, horizon_hours=1000)
    assert np.isnan(pred).all()
    print("ok  seasonal_naive: an out-of-range lookup returns NaN, not a crash")


def check_metric_row_drops_nan_predictions() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, np.nan, 29.0, 41.0])

    row = _metric_row("dummy", "alpha", 1, y_true, y_pred)
    assert (row["model"], row["location"], row["horizon"]) == ("dummy", "alpha", 1)

    expected_rmse = np.sqrt(np.mean((np.array([11, 29, 41]) - np.array([10, 30, 40])) ** 2))
    assert np.isclose(row["rmse"], expected_rmse)
    assert np.isfinite(row["rmse"]) and np.isfinite(row["mae"]) and np.isfinite(row["r2"])

    print(
        "ok  _metric_row: NaN predictions excluded from RMSE/MAE/R2 "
        "(not propagated as NaN)"
    )


def check_sarima_fit_and_forecast_tiny_series() -> None:
    """Fits on a tiny (200h) synthetic series - no network, fast - and checks
    the forecast is finite and correctly time-indexed past the training data."""
    raw = _synthetic_raw(200)
    series = raw.set_index("time")["us_aqi"].asfreq("h")
    train = series.iloc[:170]

    fitted = _fit_sarima(train)
    forecast = fitted.get_forecast(steps=20).predicted_mean

    assert len(forecast) == 20
    assert forecast.index[0] == train.index[-1] + pd.Timedelta(hours=1)
    assert np.isfinite(forecast.to_numpy()).all()

    print(
        f"ok  sarima: fits {SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} on a tiny "
        "synthetic series and forecasts forward with a finite, "
        "correctly-indexed result (no network)"
    )


def check_ml_registered_rows_delegates_to_registry() -> None:
    """``ml_registered_rows`` now delegates its ``load_best_model`` loop to
    ``registry.current_model_metrics`` - must still produce the exact same
    row shape/values as before that refactor."""
    project = _FakeProject(_FakeModelRegistry())
    registry._project = lambda: project  # swap the Hopsworks handle for a fake

    X = np.arange(20).reshape(-1, 2).astype(float)
    y = X.sum(axis=1)
    model = LinearRegression().fit(X, y)

    for horizon in evaluate_baselines.HORIZONS:
        name = evaluate_baselines.model_name(horizon)
        registry.register_model(
            name,
            model,
            {
                "val": {"rmse": 2.0, "mae": 1.5, "r2": 0.7},
                "test": {"rmse": 3.0 + horizon * 0.01, "mae": 2.0, "r2": 0.8},
                "algorithm": "linreg",
            },
            ["f0", "f1"],
        )

    rows = evaluate_baselines.ml_registered_rows()
    assert len(rows) == len(evaluate_baselines.HORIZONS)

    by_horizon = {r["horizon"]: r for r in rows}
    for horizon in evaluate_baselines.HORIZONS:
        r = by_horizon[horizon]
        assert r["model"] == "linreg (served v1)", r
        assert r["location"] == "all"
        assert np.isclose(r["rmse"], 3.0 + horizon * 0.01)
        assert r["mae"] == 2.0
        assert r["r2"] == 0.8

    print(
        "ok  ml_registered_rows: delegates to registry.current_model_metrics, "
        "same row shape/values as before the refactor"
    )


def main() -> int:
    check_seasonal_naive_predictions()
    check_seasonal_naive_missing_lookup_is_nan()
    check_metric_row_drops_nan_predictions()
    check_sarima_fit_and_forecast_tiny_series()
    check_ml_registered_rows_delegates_to_registry()
    print("\nall evaluate_baselines smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
