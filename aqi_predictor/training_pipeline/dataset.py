"""Build the supervised training frame and time-ordered splits.

The prediction target is next-hour US AQI (``us_aqi_next = us_aqi.shift(-1)``),
computed per location. Splits are strictly time-ordered (no shuffling): for each
location the last ``SPLIT_DAYS`` days are the test set, the ``SPLIT_DAYS`` days
before that are validation, and everything earlier is training.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aqi_predictor.feature_pipeline import store

TARGET = "us_aqi_next"
SOURCE_TARGET = "us_aqi"
ID_COLUMNS = ["location", "time"]
SPLIT_DAYS = 14

# Far-apart bounds so a single call pulls the whole feature store.
_FULL_START = "2000-01-01"
_FULL_END = "2100-01-01"


@dataclass
class Splits:
    """Time-ordered train / val / test frames plus the shared feature list."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]

    def xy(self, part: str) -> tuple[pd.DataFrame, pd.Series]:
        frame = getattr(self, part)
        return frame[self.feature_columns], frame[TARGET]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column that is a model input (not an id column, not the target)."""
    excluded = set(ID_COLUMNS) | {TARGET}
    return [c for c in df.columns if c not in excluded]


def build_training_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the feature store with a per-location ``us_aqi_next`` target.

    Rows with no known next value (each location's last row) and rows with any
    missing feature value (each location's first ~24h, where long-window
    features are undefined) are dropped.
    """
    if df is None:
        df = store.get_feature_view(_FULL_START, _FULL_END)
    if df.empty:
        raise RuntimeError("feature store returned no rows; run scripts/backfill.py first")

    df = df.sort_values(ID_COLUMNS).reset_index(drop=True)
    df[TARGET] = df.groupby("location", sort=False)[SOURCE_TARGET].shift(-1)

    feats = feature_columns(df)
    before = len(df)
    df = df.dropna(subset=[TARGET, *feats]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[dataset] dropped {dropped} rows (no next value / incomplete features)")
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


def split_dataset(df: pd.DataFrame | None = None) -> Splits:
    """Time-ordered per-location split into train / val / test."""
    if df is None:
        df = build_training_frame()

    df = df.sort_values(ID_COLUMNS).reset_index(drop=True)
    labels = _split_labels(df)
    feats = feature_columns(df)
    parts = {
        name: df[labels == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    return Splits(feature_columns=feats, **parts)
