"""Offline smoke checks for the dashboard layer (no pytest, no Streamlit, no network).

Run:  python tests/smoke_dashboard.py

Covers:
* ``aqi_scale.aqi_category`` maps every US AQI breakpoint (and each boundary
  value) to the right label,
* a ``forecast()``-shaped dict's ``recent`` / ``forecasts`` fields are
  well-formed - expected keys, parseable ISO timestamps.
"""

from __future__ import annotations

import sys

import pandas as pd

from aqi_predictor.aqi_scale import aqi_category

_EXPECTED_LABELS = {
    0: "Good",
    50: "Good",
    51: "Moderate",
    100: "Moderate",
    101: "Unhealthy for Sensitive Groups",
    150: "Unhealthy for Sensitive Groups",
    151: "Unhealthy",
    200: "Unhealthy",
    201: "Very Unhealthy",
    300: "Very Unhealthy",
    301: "Hazardous",
    400: "Hazardous",
}


def check_aqi_category_boundaries() -> None:
    for value, expected in _EXPECTED_LABELS.items():
        label, colour = aqi_category(value)
        assert label == expected, (value, label, expected)
        assert colour.startswith("#") and len(colour) == 7, (value, colour)

    # non-finite input is rejected, not silently bucketed
    for bad in (float("nan"), float("inf")):
        try:
            aqi_category(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")

    print("ok  aqi_category: every breakpoint + boundary maps to the right label")


def _synthetic_forecast() -> dict:
    now = pd.Timestamp("2026-09-01T08:00:00Z")
    recent = [
        {"time": (now - pd.Timedelta(hours=h)).isoformat(), "us_aqi": 60.0 + h % 5}
        for h in range(48, -1, -1)
    ]
    forecasts = [
        {
            "horizon_hours": h,
            "target_time": (now + pd.Timedelta(hours=h)).isoformat(),
            "predicted_us_aqi": 65.0 + h / 10,
            "model_name": "us_aqi_next" if h == 1 else f"us_aqi_h{h}",
            "model_version": 2,
        }
        for h in (1, 24, 48, 72)
    ]
    return {
        "location": "karachi",
        "generated_at": now.isoformat(),
        "current": {"time": now.isoformat(), "us_aqi": 61.0},
        "recent": recent,
        "forecasts": forecasts,
    }


def check_forecast_dict_shape() -> None:
    result = _synthetic_forecast()

    for key in ("location", "generated_at", "current", "recent", "forecasts"):
        assert key in result, key
    pd.Timestamp(result["generated_at"])

    assert set(result["current"]) == {"time", "us_aqi"}
    pd.Timestamp(result["current"]["time"])
    assert isinstance(result["current"]["us_aqi"], (int, float))

    assert len(result["recent"]) >= 1
    for row in result["recent"]:
        assert set(row) == {"time", "us_aqi"}, row
        pd.Timestamp(row["time"])
        assert isinstance(row["us_aqi"], (int, float))

    assert len(result["forecasts"]) == 4
    for fc in result["forecasts"]:
        assert set(fc) == {
            "horizon_hours",
            "target_time",
            "predicted_us_aqi",
            "model_name",
            "model_version",
        }, fc
        pd.Timestamp(fc["target_time"])
        assert isinstance(fc["predicted_us_aqi"], (int, float))
        assert isinstance(fc["horizon_hours"], int)

    # the fields the dashboard actually charts load into a DataFrame cleanly
    recent_df = pd.DataFrame(result["recent"])
    recent_df["time"] = pd.to_datetime(recent_df["time"])
    fc_df = pd.DataFrame(result["forecasts"])
    fc_df["target_time"] = pd.to_datetime(fc_df["target_time"])
    assert recent_df["time"].is_monotonic_increasing
    assert list(fc_df["horizon_hours"]) == [1, 24, 48, 72]

    print("ok  forecast dict: recent / forecasts well-formed, timestamps parse")


def main() -> int:
    check_aqi_category_boundaries()
    check_forecast_dict_shape()
    print("\nall dashboard smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
