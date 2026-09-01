"""Hopsworks Feature Store integration for the AQI feature pipeline.

Public API (unchanged from the earlier local-parquet implementation - every call
site keeps working without edits):

* :func:`insert_features` - upsert rows into the ``aqi_features`` feature group,
  keyed on ``(location, time)``. The feature group has a primary key, so
  Hopsworks/Hudi treats a re-inserted key as an overwrite: inserting overlapping
  rows is idempotent, no manual dedup needed.
* :func:`get_feature_view` - read the whole feature group back and return the
  rows with ``start <= time <= end`` (bounds inclusive), optionally filtered to a
  set of locations. Empty ``DataFrame`` if nothing matches; sorted by
  ``(location, time)``.

Both always talk to the real Hopsworks project configured via
``HOPSWORKS_API_KEY`` / ``HOPSWORKS_PROJECT_NAME`` - there is no local fallback.

Timestamp handling: Hopsworks' offline store does not accept a timezone-aware
timestamp column, so :func:`insert_features` strips the tz (the values are UTC)
before ``fg.insert`` and :func:`get_feature_view` re-attaches UTC on read.
Callers still pass and receive tz-aware UTC timestamps, exactly as before.

The feature group is HUDI + stream with statistics disabled. On this deployment
(Hopsworks Serverless, external Windows client) that is the only write path that
works end to end: the DELTA + delta-rs path needs a direct HDFS namenode
connection from the client (this project's offline store is HopsFS, not S3), and
the HUDI Spark job's built-in statistics step was failing with a backend 500
("Transaction marked for rollback"). ``statistics_config=False`` skips that step;
this project does not use Hopsworks-computed statistics anyway.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from aqi_predictor import hopsworks_client

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

_KEY = ["location", "time"]


def _project():
    """Hopsworks project handle. Separate function so tests can monkeypatch it."""
    return hopsworks_client.get_project()


def _feature_group():
    fs = _project().get_feature_store()
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            "Hourly engineered AQI + weather features, one row per "
            "(location, time). Written by the feature pipeline / backfill."
        ),
        primary_key=_KEY,
        event_time="time",
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config=False,
    )


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
    """Upsert ``df`` into the ``aqi_features`` feature group, keyed on
    ``(location, time)``. Rows with an existing key are overwritten.
    """
    if df is None or df.empty:
        return

    incoming = _normalise(df)
    # Hopsworks rejects a tz-aware timestamp column on the offline store; the
    # values are UTC, so drop the tz here and re-attach it on read.
    incoming["time"] = incoming["time"].dt.tz_localize(None)

    # wait=True so the rows are queryable by the time this returns (the migration
    # script and dataset.py read back immediately after).
    _feature_group().insert(incoming, wait=True)


def get_feature_view(
    start: str | dt.date | dt.datetime | pd.Timestamp,
    end: str | dt.date | dt.datetime | pd.Timestamp,
    locations: list[str] | None = None,
) -> pd.DataFrame:
    """Return stored feature rows with ``start <= time <= end`` (bounds inclusive).

    ``locations`` defaults to every location currently in the feature group.
    """
    lo, hi = _to_utc_ts(start), _to_utc_ts(end)

    raw = _feature_group().read(dataframe_type="pandas")
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    df = raw.copy()
    # tz was stripped on write (values are UTC) - put it back.
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["location"] = df["location"].astype(str)

    mask = (df["time"] >= lo) & (df["time"] <= hi)
    if locations is not None:
        mask &= df["location"].isin(list(locations))

    out = df[mask]
    if out.empty:
        return pd.DataFrame()
    return out.sort_values(_KEY).reset_index(drop=True)
