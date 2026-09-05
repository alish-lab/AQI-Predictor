"""Offline smoke checks for the dashboard layer (no pytest, no network).

Run:  python tests/smoke_dashboard.py

Covers:
* ``aqi_scale.aqi_category`` maps every US AQI breakpoint (and each boundary
  value) to the right label,
* a ``forecast()``-shaped dict's ``recent`` / ``forecasts`` fields are
  well-formed - expected keys, parseable ISO timestamps,
* the Streamlit app renders end to end against a stubbed ``predict.forecast``
  (via ``AppTest``, no network): no exception, the four stat cards carry the
  right AQI numbers + category, the details expander holds all four rows, and a
  pipeline failure is caught and shown as ``st.error`` rather than crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from aqi_predictor.aqi_scale import aqi_category

_APP_PATH = str(
    Path(__file__).resolve().parents[1] / "aqi_predictor" / "dashboard" / "app.py"
)

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


def _synthetic_forecast_with_shap() -> dict:
    """Like :func:`_synthetic_forecast`, but every horizon carries
    ``top_features`` / ``shap_importance`` shaped exactly like the LSTM path's
    (signed local values summed across time, non-negative global mean|SHAP|) -
    the dashboard's rendering code doesn't know or care which model
    architecture produced them, so this is what "covering the LSTM path"
    means for the dashboard layer.
    """
    result = _synthetic_forecast()
    feature_names = ["pm2_5_lag_1h", "temperature_2m_target", "us_aqi_roll_mean_24h"]
    for fc in result["forecasts"]:
        fc["top_features"] = [
            {"feature": name, "shap_value": (-1) ** j * (3.0 - j), "value": 10.0 + j}
            for j, name in enumerate(feature_names)
        ]
        fc["shap_importance"] = {name: 1.0 + j for j, name in enumerate(feature_names)}
    return result


def check_app_renders_with_shap() -> None:
    """The 'Explain this forecast' section renders for a forecast whose
    ``top_features`` / ``shap_importance`` came from the LSTM path - no
    dashboard code change was needed, this just verifies nothing in the
    rendering code silently assumed a tree-model-only shape."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from aqi_predictor.inference_pipeline import predict

    result = _synthetic_forecast_with_shap()

    st.cache_data.clear()
    with patch.object(predict, "forecast", return_value=result):
        at = AppTest.from_file(_APP_PATH, default_timeout=60).run()

    assert not at.exception, at.exception
    subheaders = " ".join(m.value for m in at.subheader)
    assert "Explain this forecast" in subheaders

    print(
        "ok  app: explain section renders for LSTM-shaped SHAP fields "
        "(signed local top_features, non-negative global shap_importance)"
    )


def check_app_renders() -> None:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from aqi_predictor.inference_pipeline import predict

    result = _synthetic_forecast()

    st.cache_data.clear()
    with patch.object(predict, "forecast", return_value=result):
        at = AppTest.from_file(_APP_PATH, default_timeout=60).run()

    assert not at.exception, at.exception

    # the four stat cards are unsafe-HTML markdown blocks; each carries its number
    card_html = " ".join(m.value for m in at.markdown if "aqi-card" in m.value)
    assert "Now" in card_html and "+24h" in card_html and "+72h" in card_html
    for label, value in (
        ("Now", result["current"]["us_aqi"]),
        *((f"+{f['horizon_hours']}h", f["predicted_us_aqi"]) for f in result["forecasts"][1:]),
    ):
        assert f">{value:.0f}<" in card_html, (label, value, card_html[:300])
    assert "Moderate" in card_html  # every synthetic value sits in one category

    # the details expander holds the full 4-row forecast table
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) == 4
    assert list(at.dataframe[0].value["horizon_hours"]) == [1, 24, 48, 72]

    # a pipeline failure is caught, not raised
    st.cache_data.clear()
    with patch.object(predict, "forecast", side_effect=RuntimeError("boom")):
        at_err = AppTest.from_file(_APP_PATH, default_timeout=60).run()
    assert not at_err.exception
    assert any("Could not load" in e.value for e in at_err.error)
    assert not at_err.dataframe  # render() is skipped on error

    print("ok  app: cards + charts + expander render, errors surface as st.error")


def main() -> int:
    check_aqi_category_boundaries()
    check_forecast_dict_shape()
    check_app_renders()
    check_app_renders_with_shap()
    print("\nall dashboard smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
