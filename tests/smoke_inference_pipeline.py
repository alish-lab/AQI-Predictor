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

from aqi_predictor.inference_pipeline.predict import build_input_row, model_name

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


def main() -> int:
    check_model_name()
    check_columns_match_feature_list()
    check_target_columns_use_forecast()
    check_missing_feature_raises()
    check_uncovered_target_time_raises()
    print("\nall inference-pipeline smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
