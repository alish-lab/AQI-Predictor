"""Offline smoke checks for aqi_predictor/dashboard/eda_charts.py (no network).

Run:  python tests/smoke_eda_charts.py

Doesn't check exact chart contents - just that each of the four shared chart
functions runs without error against a small synthetic dataframe and returns
a renderable Altair chart (``.to_dict()`` forces Altair's own validation).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from aqi_predictor.dashboard import eda_charts


def _synthetic_history(n_hours: int = 24 * 14) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    base = 70 + 20 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    df = pd.DataFrame(
        {
            "location": "alpha",
            "time": idx,
            "us_aqi": base + rng.normal(0, 5, n_hours),
            "pm10": 50 + rng.normal(0, 5, n_hours),
            "pm2_5": 30 + rng.normal(0, 5, n_hours),
            "carbon_monoxide": 200 + rng.normal(0, 10, n_hours),
            "nitrogen_dioxide": 20 + rng.normal(0, 2, n_hours),
            "sulphur_dioxide": 5 + rng.normal(0, 1, n_hours),
            "ozone": 40 + rng.normal(0, 3, n_hours),
            "temperature_2m": 25 + rng.normal(0, 3, n_hours),
            "relative_humidity_2m": 50 + rng.normal(0, 5, n_hours),
            "wind_speed_10m": 10 + rng.normal(0, 2, n_hours),
            "wind_direction_10m": 180 + rng.normal(0, 30, n_hours),
            "surface_pressure": 1000 + rng.normal(0, 2, n_hours),
            "precipitation": rng.exponential(0.1, n_hours),
        }
    )
    # a handful of forced "Hazardous" hours so category_distribution has a
    # non-empty bar for every category, and a dropped run so trend_and_gaps
    # actually has a gap to shade.
    df.loc[5:8, "us_aqi"] = 350.0
    return df.drop(df.index[100:104]).reset_index(drop=True)


def check_trend_and_gaps() -> None:
    chart = eda_charts.trend_and_gaps(_synthetic_history())
    chart.to_dict()  # forces Altair's own schema validation
    print("ok  trend_and_gaps: runs on synthetic data with a real gap, no crash")


def check_trend_and_gaps_large_dataset() -> None:
    """Above eda_charts._MAX_TREND_POINTS the line is resampled to daily mean -
    exercise that branch so a multi-year history (Altair's default 5000-row
    embed limit; the real feature store is ~18k rows) doesn't blow up."""
    large = _synthetic_history(n_hours=24 * 800)  # well past _MAX_TREND_POINTS
    chart = eda_charts.trend_and_gaps(large)
    spec = chart.to_dict()
    assert len(str(spec)) < 200_000, "resample branch should keep the spec small"
    print(
        "ok  trend_and_gaps: a multi-year history is resampled to daily mean "
        "for the line (gap detection still runs hourly), spec stays small"
    )


def check_seasonality_patterns() -> None:
    chart = eda_charts.seasonality_patterns(_synthetic_history())
    chart.to_dict()
    print("ok  seasonality_patterns: runs on synthetic data, no crash")


def check_category_distribution() -> None:
    chart = eda_charts.category_distribution(_synthetic_history())
    chart.to_dict()
    print("ok  category_distribution: runs on synthetic data (incl. a forced "
          "Hazardous run), no crash")


def check_feature_correlation() -> None:
    chart = eda_charts.feature_correlation(_synthetic_history())
    chart.to_dict()
    print("ok  feature_correlation: runs on synthetic data, no crash")


def main() -> int:
    check_trend_and_gaps()
    check_trend_and_gaps_large_dataset()
    check_seasonality_patterns()
    check_category_distribution()
    check_feature_correlation()
    print("\nall eda_charts smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
