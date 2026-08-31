"""Feature engineering + missing-data handling for the AQI feature pipeline.

Input: the tidy hourly frame produced by :mod:`aqi_predictor.feature_pipeline.fetch`
(``location``, ``time``, pollutants, weather).

Output: the same rows plus engineered columns, after:

1. reindexing each location to a gap-free hourly grid,
2. linearly interpolating gaps of <= ``MAX_INTERPOLATION_GAP`` hours,
3. dropping rows where a required column is still missing.

All engineered features are backward-looking (lags, trailing rolling windows,
backward diffs) so a row never sees its own future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aqi_predictor.config import POLLUTANT_VARS, REQUIRED_COLUMNS, WEATHER_VARS

MAX_INTERPOLATION_GAP = 3  # hours

# Columns that lag / rolling / change-rate features are built from.
_SERIES_COLS = ["us_aqi", "pm2_5", "pm10"]
_LAGS = [1, 2, 3]
_ROLL_WINDOWS = [3, 6, 24]

_NUMERIC_COLS = [*POLLUTANT_VARS, *WEATHER_VARS]


@dataclass
class CleaningReport:
    """Per-location data-quality summary for one backfill/feature run."""

    location: str
    n_input_rows: int = 0
    n_hourly_grid_rows: int = 0
    n_output_rows: int = 0
    n_rows_dropped: int = 0
    time_min: pd.Timestamp | None = None
    time_max: pd.Timestamp | None = None
    missing_pct_before: dict[str, float] = field(default_factory=dict)
    missing_pct_after: dict[str, float] = field(default_factory=dict)
    cells_interpolated: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "n_input_rows": self.n_input_rows,
            "n_hourly_grid_rows": self.n_hourly_grid_rows,
            "n_output_rows": self.n_output_rows,
            "n_rows_dropped": self.n_rows_dropped,
            "n_cells_interpolated_total": int(sum(self.cells_interpolated.values())),
            "time_min": None if self.time_min is None else str(self.time_min),
            "time_max": None if self.time_max is None else str(self.time_max),
            "missing_pct_before": self.missing_pct_before,
            "missing_pct_after": self.missing_pct_after,
            "cells_interpolated": self.cells_interpolated,
        }


# --------------------------------------------------------------------------- #
# Missing-data handling
# --------------------------------------------------------------------------- #
def _to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a single-location frame onto a complete hourly UTC grid."""
    out = df.sort_values("time").drop_duplicates(subset="time", keep="last").copy()
    full = pd.date_range(out["time"].min(), out["time"].max(), freq="h", tz="UTC")
    out = out.set_index("time").reindex(full)
    out.index.name = "time"
    out["location"] = out["location"].ffill().bfill()
    return out.reset_index()


def _missing_pct(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    n = len(df)
    if n == 0:
        return {c: 0.0 for c in cols}
    return {c: round(100.0 * df[c].isna().mean(), 3) for c in cols}


def _interpolate_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate only runs of <= ``max_gap`` consecutive NaNs.

    Longer gaps are left untouched (their rows get dropped later if the column
    is required). Leading/trailing gaps are never filled - there is nothing to
    interpolate between.
    """
    na = s.isna()
    if not na.any():
        return s
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")  # NaN-run length; 0 elsewhere
    filled = s.interpolate(method="linear", limit_area="inside")
    return filled.mask(na & (run_len > max_gap), other=np.nan)


# --------------------------------------------------------------------------- #
# Engineered features
# --------------------------------------------------------------------------- #
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["time"].dt
    df["hour"] = t.hour
    df["day_of_week"] = t.dayofweek
    df["month"] = t.month
    df["is_weekend"] = (t.dayofweek >= 5).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def _add_change_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    df["us_aqi_change_1h"] = df["us_aqi"].diff(1)
    df["us_aqi_change_24h"] = df["us_aqi"].diff(24)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in _SERIES_COLS:
        for lag in _LAGS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in _SERIES_COLS:
        for w in _ROLL_WINDOWS:
            roll = df[col].rolling(window=w, min_periods=w)
            df[f"{col}_roll_mean_{w}h"] = roll.mean()
            df[f"{col}_roll_std_{w}h"] = roll.std()
    return df


def _engineer_one(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    location = str(df["location"].iloc[0]) if len(df) else "unknown"
    report = CleaningReport(location=location, n_input_rows=len(df))

    grid = _to_hourly_grid(df)
    report.n_hourly_grid_rows = len(grid)

    # Features are computed on the gap-free grid (post-interpolation) so lags and
    # rolling windows line up with real clock hours; rows still missing a
    # required column are dropped afterwards.
    present = [c for c in _NUMERIC_COLS if c in grid.columns]
    report.missing_pct_before = _missing_pct(grid, present)

    before_na = grid[present].isna()
    for col in present:
        grid[col] = _interpolate_short_gaps(grid[col], MAX_INTERPOLATION_GAP)
    after_na = grid[present].isna()
    report.cells_interpolated = {
        c: int((before_na[c] & ~after_na[c]).sum()) for c in present
    }

    grid = _add_time_features(grid)
    grid = _add_change_rate_features(grid)
    grid = _add_lag_features(grid)
    grid = _add_rolling_features(grid)

    required = [c for c in REQUIRED_COLUMNS if c in grid.columns]
    keep = grid[required].notna().all(axis=1)
    report.n_rows_dropped = int((~keep).sum())
    out = grid[keep].reset_index(drop=True)

    lead = ["location", "time"]
    out = out[lead + [c for c in out.columns if c not in lead]]

    report.missing_pct_after = _missing_pct(out, present)
    report.n_output_rows = len(out)
    if len(out):
        report.time_min = out["time"].min()
        report.time_max = out["time"].max()
    return out, report


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[CleaningReport]]:
    """Engineer features + handle missing data, per location.

    Returns the combined feature frame and one :class:`CleaningReport` per
    location.
    """
    if df.empty:
        return df.copy(), []

    frames: list[pd.DataFrame] = []
    reports: list[CleaningReport] = []
    for _, group in df.groupby("location", sort=True):
        out, report = _engineer_one(group)
        frames.append(out)
        reports.append(report)

    combined = pd.concat(frames, ignore_index=True) if frames else df.copy()
    return combined, reports
