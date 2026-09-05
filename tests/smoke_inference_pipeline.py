"""Offline smoke checks for the inference pipeline (no network, no pytest).

Run:  python tests/smoke_inference_pipeline.py

Verifies that ``predict.build_input_row``:
* produces a row whose columns are *exactly* the model's ``feature_list``, in the
  right order (extra columns on the "now" row are dropped),
* fills each ``<var>_target`` column from the forecast value for that horizon -
  the future weather, not the "now" value,
* raises a clear error on a missing feature or an uncovered target time, rather
  than silently handing the model a mismatched row.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from aqi_predictor.inference_pipeline.predict import (
    _local_shap_top_features,
    _lstm_local_shap,
    build_input_row,
    build_input_sequence,
    model_name,
)
from aqi_predictor.training_pipeline.lstm_model import AQI_LSTM

NOW = pd.Timestamp("2025-06-01T12:00:00Z")


def _now_row() -> pd.Series:
    # temperature_2m / wind_speed_10m are deliberately absurd here: if a
    # <var>_target column ever picked them up instead of the forecast, the
    # assertions below would catch it.
    return pd.Series(
        {
            "location": "testville",
            "time": NOW,
            "us_aqi": 90.0,
            "pm2_5": 30.0,
            "pm10": 55.0,
            "hour": 12.0,
            "temperature_2m": 999.0,
            "relative_humidity_2m": 999.0,
            "wind_speed_10m": 999.0,
            "wind_direction_10m": 999.0,
            "surface_pressure": 999.0,
            "precipitation": 999.0,
            "some_unused_feature": 123.0,
        }
    )


def _forecast_weather() -> pd.DataFrame:
    times = pd.date_range(
        NOW - pd.Timedelta(hours=2), NOW + pd.Timedelta(hours=90), freq="h", tz="UTC"
    )
    off = (times - NOW).total_seconds() / 3600.0  # hours from NOW (can be negative)
    return pd.DataFrame(
        {
            "time": times,
            "temperature_2m": 20.0 + off,
            "relative_humidity_2m": 50.0 + off,
            "wind_speed_10m": 10.0 + off,
            "wind_direction_10m": 100.0 + off,
            "surface_pressure": 1000.0 + off,
            "precipitation": off / 10.0,
        }
    )


def _observed_window(n: int = 6) -> pd.DataFrame:
    """``n`` hourly rows of engineered features ending at ``NOW``, like the
    trailing slice of ``predict._current_and_history``'s ``observed`` frame."""
    times = pd.date_range(NOW - pd.Timedelta(hours=n - 1), NOW, freq="h", tz="UTC")
    off = (times - NOW).total_seconds() / 3600.0
    return pd.DataFrame(
        {
            "time": times,
            "pm2_5": 30.0 + off,
            "us_aqi": 90.0 + off,
            "hour": times.hour.astype(float),
            # deliberately absurd, like _now_row(): a <var>_target column that
            # ever picked these up instead of the forecast would be caught below
            "temperature_2m": 999.0,
            "relative_humidity_2m": 999.0,
            "wind_speed_10m": 999.0,
            "wind_direction_10m": 999.0,
            "surface_pressure": 999.0,
            "precipitation": 999.0,
        }
    )


def check_build_input_sequence() -> None:
    window = _observed_window(6)
    fc = _forecast_weather()
    feature_list = ["pm2_5", "us_aqi", "hour", "temperature_2m_target", "wind_speed_10m_target"]
    horizon = 24

    seq = build_input_sequence(window, feature_list, fc, horizon)
    assert list(seq.columns) == feature_list
    assert seq.shape == (len(window), len(feature_list))

    # each row's own <var>_target is the forecast value at *that row's* time +
    # horizon - not the window's last row's target time, and never the "now" value
    for i, row in window.reset_index(drop=True).iterrows():
        target_time = row["time"] + pd.Timedelta(hours=horizon)
        expected = 20.0 + (target_time - NOW).total_seconds() / 3600.0
        got = seq.loc[i, "temperature_2m_target"]
        assert np.isclose(got, expected), (i, got, expected)
        assert not np.isclose(got, 999.0)
    assert seq["pm2_5"].tolist() == window["pm2_5"].tolist()  # non-target: copied as-is

    print(
        "ok  build_input_sequence: per-row <var>_target uses that row's own "
        "time + horizon, non-target columns copied as-is"
    )


def check_build_input_sequence_missing_feature_raises() -> None:
    window = _observed_window(4)
    fc = _forecast_weather()
    try:
        build_input_sequence(window, ["does_not_exist"], fc, 24)
    except RuntimeError as exc:
        assert "does_not_exist" in str(exc), str(exc)
    else:
        raise AssertionError("expected RuntimeError for a missing feature")
    print("ok  build_input_sequence: missing feature raises a clear error")


def check_lstm_local_shap() -> None:
    rng = np.random.default_rng(0)
    seq_len, n_features = 6, 3
    feature_columns = ["f0", "f1", "f2"]

    model = AQI_LSTM(n_features=n_features, hidden_size=8, seq_len=seq_len)
    model.eval()

    background = rng.normal(size=(10, seq_len, n_features)).astype("float32")
    x_scaled = torch.tensor(rng.normal(size=(1, seq_len, n_features)).astype("float32"))
    raw_last_row = pd.Series({"f0": 1.0, "f1": 2.0, "f2": 3.0})

    top = _lstm_local_shap(model, x_scaled, background, feature_columns, raw_last_row)
    assert top is not None
    assert 1 <= len(top) <= len(feature_columns)
    assert all({"feature", "shap_value", "value"} == set(item) for item in top)
    assert {item["feature"] for item in top} <= set(feature_columns)
    abs_values = [abs(item["shap_value"]) for item in top]
    assert abs_values == sorted(abs_values, reverse=True)  # descending by |shap_value|

    assert _lstm_local_shap(model, x_scaled, None, feature_columns, raw_last_row) is None

    print(
        "ok  lstm local shap: GradientExplainer -> ranked top features, "
        "no background -> None (no crash)"
    )


def check_model_name() -> None:
    assert model_name(1) == "us_aqi_next"
    assert model_name(24) == "us_aqi_h24"
    assert model_name(72) == "us_aqi_h72"
    print("ok  model_name: h=1 -> us_aqi_next, else us_aqi_h<h>")


def check_columns_match_feature_list() -> None:
    now, fc = _now_row(), _forecast_weather()
    feature_list = [
        "pm2_5",
        "us_aqi",
        "hour",
        "wind_speed_10m_target",
        "temperature_2m_target",
    ]
    row = build_input_row(now, feature_list, fc, NOW + pd.Timedelta(hours=24))

    assert list(row.columns) == feature_list, list(row.columns)
    assert row.shape == (1, len(feature_list))
    assert "some_unused_feature" not in row.columns  # extra "now" columns dropped
    assert row["us_aqi"].iloc[0] == 90.0             # non-target copied from now
    print("ok  input row: columns == feature_list exactly, in order; extras dropped")


def check_target_columns_use_forecast() -> None:
    now, fc = _now_row(), _forecast_weather()
    for h in (1, 24, 48, 72):
        row = build_input_row(
            now, ["temperature_2m_target"], fc, NOW + pd.Timedelta(hours=h)
        )
        got = row["temperature_2m_target"].iloc[0]
        assert np.isclose(got, 20.0 + h), (h, got)   # forecast value at NOW + h
        assert not np.isclose(got, 999.0)            # never the "now" value
    print("ok  <var>_target: takes the forecast value for NOW + horizon, not now")


def check_missing_feature_raises() -> None:
    now, fc = _now_row(), _forecast_weather()
    tt = NOW + pd.Timedelta(hours=24)

    for bad_list, needle in (
        (["us_aqi", "does_not_exist"], "does_not_exist"),
        (["ozone_target"], "ozone_target"),  # base "ozone" is not a forecast weather var
    ):
        try:
            build_input_row(now, bad_list, fc, tt)
        except RuntimeError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"expected RuntimeError for {bad_list}")
    print("ok  missing feature: raises a clear error naming the column")


def check_uncovered_target_time_raises() -> None:
    now, fc = _now_row(), _forecast_weather()
    try:
        build_input_row(
            now, ["temperature_2m_target"], fc, NOW + pd.Timedelta(hours=500)
        )
    except RuntimeError as exc:
        assert "does not cover" in str(exc), str(exc)
    else:
        raise AssertionError("expected RuntimeError for an uncovered target time")
    print("ok  uncovered target time: raises a clear error")


def check_local_shap_top_features() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"f0": rng.normal(size=40), "f1": rng.normal(size=40), "f2": rng.normal(size=40)}
    )
    y = X["f0"] * 3 - X["f1"] * 2 + X["f2"] * 0.1
    row = X.iloc[[0]]

    tree_model = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
    top = _local_shap_top_features(tree_model, row)
    assert top is not None
    assert 1 <= len(top) <= 3
    assert all({"feature", "shap_value", "value"} == set(item) for item in top)
    assert {item["feature"] for item in top} <= {"f0", "f1", "f2"}
    # descending by |shap_value|
    abs_values = [abs(item["shap_value"]) for item in top]
    assert abs_values == sorted(abs_values, reverse=True)
    # f0/f1 dominate the target (coefficients 3 and -2) vs f2 (0.1) -> top-ranked
    # feature must be one of the two large-coefficient ones, not the noise feature
    assert top[0]["feature"] in {"f0", "f1"}

    linear_model = LinearRegression().fit(X, y)
    assert _local_shap_top_features(linear_model, row) is None  # non-tree -> no crash
    print("ok  local shap: tree model -> ranked top features, non-tree model -> None (no crash)")


def main() -> int:
    check_model_name()
    check_columns_match_feature_list()
    check_target_columns_use_forecast()
    check_missing_feature_raises()
    check_uncovered_target_time_raises()
    check_local_shap_top_features()
    check_build_input_sequence()
    check_build_input_sequence_missing_feature_raises()
    check_lstm_local_shap()
    print("\nall inference-pipeline smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
