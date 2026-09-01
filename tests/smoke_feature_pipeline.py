"""Offline smoke checks for the feature pipeline (no network, no pytest).

Run:  python tests/smoke_feature_pipeline.py

Guards the two invariants from the Phase 1 definition of done:
* engineered features never see a row's own future (no leakage),
* missing-data handling interpolates short gaps and drops rows still missing a
  required column,
and that the feature store round-trips a time slice (against an in-memory fake
Hopsworks feature group - no network).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from aqi_predictor.feature_pipeline import store
from aqi_predictor.feature_pipeline.features import build_features


# --------------------------------------------------------------------------- #
# In-memory stand-in for the Hopsworks feature group (no network in the tests).
# Mirrors the upsert-by-primary-key semantics the real feature group has, so
# check_store_roundtrip still exercises store.insert_features / get_feature_view.
# --------------------------------------------------------------------------- #
class _FakeFeatureGroup:
    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None

    def insert(self, df: pd.DataFrame, **_kw) -> None:
        incoming = df.copy()
        combined = incoming if self._df is None else pd.concat(
            [self._df, incoming], ignore_index=True
        )
        self._df = (
            combined.drop_duplicates(subset=["location", "time"], keep="last")
            .reset_index(drop=True)
        )

    def read(self, **_kw) -> pd.DataFrame:
        return self._df.copy() if self._df is not None else pd.DataFrame()


class _FakeFeatureStore:
    def __init__(self, fg: _FakeFeatureGroup) -> None:
        self._fg = fg

    def get_or_create_feature_group(self, *_a, **_kw) -> _FakeFeatureGroup:
        return self._fg


class _FakeProject:
    def __init__(self, fs: _FakeFeatureStore) -> None:
        self._fs = fs

    def get_feature_store(self) -> _FakeFeatureStore:
        return self._fs


def _synthetic(n_hours: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    base = 80 + 20 * np.sin(np.arange(n_hours) / 12)
    return pd.DataFrame(
        {
            "location": "testville",
            "time": idx,
            "pm10": base + rng.normal(0, 3, n_hours),
            "pm2_5": base / 2 + rng.normal(0, 2, n_hours),
            "carbon_monoxide": rng.normal(150, 10, n_hours),
            "nitrogen_dioxide": rng.normal(10, 2, n_hours),
            "sulphur_dioxide": rng.normal(5, 1, n_hours),
            "ozone": rng.normal(40, 5, n_hours),
            "us_aqi": np.round(base).astype(int),
            "temperature_2m": rng.normal(25, 3, n_hours),
            "relative_humidity_2m": rng.normal(60, 10, n_hours),
            "wind_speed_10m": rng.normal(12, 4, n_hours),
            "wind_direction_10m": rng.normal(180, 40, n_hours),
            "surface_pressure": rng.normal(1008, 3, n_hours),
            "precipitation": np.zeros(n_hours),
        }
    )


def check_no_leakage() -> None:
    raw = _synthetic()
    feat, _ = build_features(raw)
    feat = feat.sort_values("time").reset_index(drop=True)

    for i in (50, 120, 200):
        assert feat.loc[i, "us_aqi_lag_1h"] == feat.loc[i - 1, "us_aqi"]
        assert feat.loc[i, "us_aqi_lag_3h"] == feat.loc[i - 3, "us_aqi"]
        assert feat.loc[i, "us_aqi_change_24h"] == (
            feat.loc[i, "us_aqi"] - feat.loc[i - 24, "us_aqi"]
        )
        expected = feat.loc[i - 2 : i, "us_aqi"].mean()
        assert abs(feat.loc[i, "us_aqi_roll_mean_3h"] - expected) < 1e-9
    print("ok  no-leakage: lags / diffs / rolling windows are backward-looking")


def check_missing_data() -> None:
    raw = _synthetic()
    # 2-hour gap in a required column -> should be interpolated
    raw.loc[100:101, "us_aqi"] = np.nan
    # 5-hour gap in a required column -> should be dropped
    raw.loc[150:154, "pm2_5"] = np.nan

    feat, reports = build_features(raw)
    rep = reports[0]
    assert rep.cells_interpolated["us_aqi"] == 2, rep.cells_interpolated
    assert rep.n_rows_dropped == 5, rep.n_rows_dropped
    assert not feat["us_aqi"].isna().any()
    assert not feat["pm2_5"].isna().any()
    print("ok  missing-data: <=3h gap interpolated, >3h gap dropped")


def check_store_roundtrip() -> None:
    raw = _synthetic()
    feat, _ = build_features(raw)

    project = _FakeProject(_FakeFeatureStore(_FakeFeatureGroup()))
    store._project = lambda: project  # swap the Hopsworks handle for a fake

    store.insert_features(feat)
    store.insert_features(feat)  # idempotent (upsert by (location, time))

    full = store.get_feature_view("2024-01-01", "2030-01-01")
    assert len(full) == len(feat), (len(full), len(feat))
    assert not full.duplicated(["location", "time"]).any()
    assert str(full["time"].dt.tz) == "UTC"

    sl = store.get_feature_view("2025-01-03", "2025-01-04", locations=["testville"])
    assert sl["time"].min() >= pd.Timestamp("2025-01-03", tz="UTC")
    assert sl["time"].max() <= pd.Timestamp("2025-01-04 23:59", tz="UTC")
    assert len(sl) > 0

    empty = store.get_feature_view("2000-01-01", "2000-01-02")
    assert empty.empty
    print("ok  store: upsert idempotent, time slice bounded and non-empty")


def main() -> int:
    check_no_leakage()
    check_missing_data()
    check_store_roundtrip()
    print("\nall feature-pipeline smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
