"""Local feature-store fallback with a Hopsworks-shaped interface.

A later phase swaps the internals of these two functions for real Hopsworks
feature-group / feature-view calls without touching any call site:

* :func:`insert_features` - upsert rows, keyed on ``(location, time)``.
* :func:`get_feature_view` - read back a time range, optionally by location.

Storage: one Parquet file per location under
``data/feature_store/location=<name>/data.parquet``. Re-running an insert with
overlapping rows is idempotent (last write wins per key).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from aqi_predictor.config import FEATURE_STORE_DIR

_KEY = ["location", "time"]


def _partition_path(location: str) -> Path:
    return FEATURE_STORE_DIR / f"location={location}" / "data.parquet"


def _to_utc_ts(value: str | dt.date | dt.datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _KEY if c not in df.columns]
    if missing:
        raise ValueError(f"feature frame missing key column(s): {missing}")
    out = df.copy()
    out["location"] = out["location"].astype(str)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def insert_features(df: pd.DataFrame) -> None:
    """Upsert ``df`` into the feature store, keyed on ``(location, time)``.

    Existing rows with the same key are overwritten by the incoming values.
    """
    if df is None or df.empty:
        return

    incoming = _normalise(df)
    for location, group in incoming.groupby("location", sort=True):
        path = _partition_path(location)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            combined = pd.concat([pd.read_parquet(path), group], ignore_index=True)
        else:
            combined = group

        combined = (
            combined.drop_duplicates(subset=_KEY, keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )
        combined.to_parquet(path, index=False)


def get_feature_view(
    start: str | dt.date | dt.datetime | pd.Timestamp,
    end: str | dt.date | dt.datetime | pd.Timestamp,
    locations: list[str] | None = None,
) -> pd.DataFrame:
    """Return stored feature rows with ``start <= time <= end`` (bounds inclusive).

    ``locations`` defaults to every location currently in the store.
    """
    lo, hi = _to_utc_ts(start), _to_utc_ts(end)

    if locations is None:
        locations = sorted(
            p.name.split("=", 1)[1]
            for p in FEATURE_STORE_DIR.glob("location=*")
            if p.is_dir()
        )

    frames: list[pd.DataFrame] = []
    for location in locations:
        path = _partition_path(location)
        if not path.exists():
            continue
        part = pd.read_parquet(path)
        part["time"] = pd.to_datetime(part["time"], utc=True)
        frames.append(part[(part["time"] >= lo) & (part["time"] <= hi)])

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(_KEY)
        .reset_index(drop=True)
    )
