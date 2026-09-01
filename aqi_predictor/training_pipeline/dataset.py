"""Build the supervised training frame and time-ordered splits.

The prediction target is US AQI ``horizon_hours`` into the future
(``us_aqi.shift(-horizon_hours)``), computed per location. ``horizon_hours=1``
(the default) keeps the legacy target name ``us_aqi_next``; larger horizons are
named ``us_aqi_h<h>`` (e.g. ``us_aqi_h24``).

Alongside the target, each weather variable's value *at target time* is added as
a ``<var>_target`` input feature (same per-location forward shift as the target).
Weather at t+h drives pollutant dispersion, so a model predicting AQI that far
ahead needs it; without it, longer horizons had no future-weather signal at all.

Splits are strictly time-ordered (no shuffling): for each location the last
``SPLIT_DAYS`` days are the test set, the ``SPLIT_DAYS`` days before that are
validation, and everything earlier is training. Larger horizons naturally drop
more rows at the tail of the series (no ground truth that far ahead yet).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aqi_predictor.config import WEATHER_VARS
from aqi_predictor.feature_pipeline import store

TARGET = "us_aqi_next"          # legacy name for the horizon_hours=1 target
SOURCE_TARGET = "us_aqi"
ID_COLUMNS = ["location", "time"]
SPLIT_DAYS = 14
DEFAULT_HORIZON_HOURS = 1

# Far-apart bounds so a single call pulls the whole feature store.
_FULL_START = "2000-01-01"
_FULL_END = "2100-01-01"


def target_name(horizon_hours: int = DEFAULT_HORIZON_HOURS) -> str:
    """Target column name for a horizon. ``h=1`` -> ``us_aqi_next`` (back-compat)."""
    if horizon_hours < 1:
        raise ValueError(f"horizon_hours must be >= 1, got {horizon_hours}")
    return TARGET if horizon_hours == 1 else f"us_aqi_h{horizon_hours}"


@dataclass
class Splits:
    """Time-ordered train / val / test frames plus the shared feature list."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target: str = TARGET

    def xy(self, part: str) -> tuple[pd.DataFrame, pd.Series]:
        frame = getattr(self, part)
        return frame[self.feature_columns], frame[self.target]


def feature_columns(df: pd.DataFrame, target: str = TARGET) -> list[str]:
    """Every column that is a model input (not an id column, not the target)."""
    excluded = set(ID_COLUMNS) | {target}
    return [c for c in df.columns if c not in excluded]


def build_training_frame(
    df: pd.DataFrame | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> pd.DataFrame:
    """Return the feature store with a per-location future-``us_aqi`` target
    plus ``<var>_target`` weather-at-target-time input features.

    Rows with no known target (each location's last ``horizon_hours`` rows) and
    rows with any missing feature value (each location's first ~24h, where
    long-window features are undefined) are dropped. The ``<var>_target`` columns
    are NaN for the same tail rows the target is, so no special-casing is needed.
    """
    tgt = target_name(horizon_hours)
    if df is None:
        df = store.get_feature_view(_FULL_START, _FULL_END)
    if df.empty:
        raise RuntimeError("feature store returned no rows; run scripts/backfill.py first")

    df = df.sort_values(ID_COLUMNS).reset_index(drop=True)
    by_location = df.groupby("location", sort=False)
    df[tgt] = by_location[SOURCE_TARGET].shift(-horizon_hours)
    for col in WEATHER_VARS:
        if col in df.columns:
            df[f"{col}_target"] = by_location[col].shift(-horizon_hours)

    feats = feature_columns(df, tgt)
    before = len(df)
    df = df.dropna(subset=[tgt, *feats]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(
            f"[dataset] horizon={horizon_hours}h: dropped {dropped} rows "
            f"(target unavailable / incomplete features)"
        )
    return df


def _split_labels(df: pd.DataFrame) -> pd.Series:
    """Per-location: last SPLIT_DAYS -> test, prior SPLIT_DAYS -> val, rest -> train."""
    max_time = df.groupby("location")["time"].transform("max")
    test_start = max_time - pd.Timedelta(days=SPLIT_DAYS)
    val_start = max_time - pd.Timedelta(days=2 * SPLIT_DAYS)

    labels = pd.Series("train", index=df.index, dtype=object)
    labels[df["time"] >= val_start] = "val"
    labels[df["time"] >= test_start] = "test"
    return labels


def split_dataset(
    df: pd.DataFrame | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> Splits:
    """Time-ordered per-location split into train / val / test.

    If ``df`` is given it must already carry the target for ``horizon_hours``
    (i.e. come from ``build_training_frame(..., horizon_hours)``).
    """
    tgt = target_name(horizon_hours)
    if df is None:
        df = build_training_frame(horizon_hours=horizon_hours)

    df = df.sort_values(ID_COLUMNS).reset_index(drop=True)
    labels = _split_labels(df)
    feats = feature_columns(df, tgt)
    parts = {
        name: df[labels == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    return Splits(feature_columns=feats, target=tgt, **parts)
