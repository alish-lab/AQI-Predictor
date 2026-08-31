"""Offline smoke checks for the training pipeline (no network, no pytest).

Run:  python tests/smoke_training_pipeline.py

Covers the Phase 2 / 2b definition-of-done invariants:
* the ``us_aqi_next`` target is a genuine future value (no leakage) and each
  location's last row is dropped,
* the horizon-parameterized target (``horizon_hours=24``) is the real value 24h
  later, and ``split_dataset`` works at a non-default horizon,
* the time-ordered train / val / test split has no overlap,
* the local model registry round-trips a model + metadata and picks "best" by
  test RMSE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from aqi_predictor.training_pipeline import dataset, registry
from aqi_predictor.training_pipeline.dataset import (
    SPLIT_DAYS,
    TARGET,
    build_training_frame,
    split_dataset,
    target_name,
)


def _synthetic(n_days: int = 60, locations=("alpha", "beta")) -> pd.DataFrame:
    n = n_days * 24
    rng = np.random.default_rng(0)
    frames = []
    for k, loc in enumerate(locations):
        idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        base = 80 + 10 * k + 15 * np.sin(np.arange(n) / 12) + rng.normal(0, 2, n)
        frames.append(
            pd.DataFrame(
                {
                    "location": loc,
                    "time": idx,
                    "us_aqi": np.round(base).astype(int),
                    "pm2_5": base / 2 + rng.normal(0, 1, n),
                    "pm10": base + rng.normal(0, 2, n),
                    "temperature_2m": rng.normal(25, 3, n),
                    "hour_sin": np.sin(2 * np.pi * idx.hour.to_numpy() / 24),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def check_target_no_leakage() -> None:
    raw = _synthetic()
    frame = build_training_frame(raw)

    for loc, g in frame.groupby("location"):
        g = g.sort_values("time").reset_index(drop=True)
        # target at row i is the *actual* us_aqi one hour later
        merged = g.merge(
            g[["time", "us_aqi"]].rename(columns={"time": "t_next", "us_aqi": "u_next"}),
            left_on=g["time"] + pd.Timedelta(hours=1),
            right_on="t_next",
            how="inner",
        )
        assert (merged[TARGET] == merged["u_next"]).all(), loc

    # each location's final row (no known next value) is gone
    raw_last = raw.groupby("location")["time"].max()
    kept_last = frame.groupby("location")["time"].max()
    assert (kept_last < raw_last).all(), (kept_last.to_dict(), raw_last.to_dict())
    assert not frame[TARGET].isna().any()
    print("ok  target: us_aqi_next is a true +1h value, last row dropped per location")


def check_horizon_target_no_leakage() -> None:
    horizon = 24
    tgt = target_name(horizon)
    assert tgt == "us_aqi_h24", tgt

    raw = _synthetic()
    frame = build_training_frame(raw, horizon_hours=horizon)

    for loc, g in frame.groupby("location"):
        g = g.sort_values("time").reset_index(drop=True)
        # target at time t must be the actual us_aqi exactly `horizon` hours later
        merged = g.merge(
            g[["time", "us_aqi"]].rename(
                columns={"time": "t_future", "us_aqi": "u_future"}
            ),
            left_on=g["time"] + pd.Timedelta(hours=horizon),
            right_on="t_future",
            how="inner",
        )
        assert len(merged) > 0
        assert (merged[tgt] == merged["u_future"]).all(), loc

    # the last `horizon` rows per location (no ground truth) are dropped
    raw_last = raw.groupby("location")["time"].max()
    kept_last = frame.groupby("location")["time"].max()
    assert (
        (raw_last - kept_last) >= pd.Timedelta(hours=horizon)
    ).all(), (kept_last.to_dict(), raw_last.to_dict())
    assert not frame[tgt].isna().any()
    print("ok  horizon target: us_aqi_h24 is the real value 24h later, tail dropped")


def check_split_non_default_horizon() -> None:
    horizon = 24
    tgt = target_name(horizon)
    frame = build_training_frame(_synthetic(), horizon_hours=horizon)
    splits = split_dataset(frame, horizon_hours=horizon)

    assert splits.target == tgt
    assert tgt not in splits.feature_columns
    assert len(splits.train) and len(splits.val) and len(splits.test)
    for part in (splits.train, splits.val, splits.test):
        assert tgt in part.columns and not part[tgt].isna().any()

    for loc in splits.test["location"].unique():
        tr = splits.train.loc[splits.train["location"] == loc, "time"]
        va = splits.val.loc[splits.val["location"] == loc, "time"]
        te = splits.test.loc[splits.test["location"] == loc, "time"]
        assert tr.max() < va.min() < va.max() < te.min(), loc
    print("ok  split @ horizon=24: target us_aqi_h24, train < val < test, no overlap")


def check_split_no_overlap() -> None:
    splits = split_dataset(build_training_frame(_synthetic()))
    assert len(splits.train) and len(splits.val) and len(splits.test)
    assert TARGET not in splits.feature_columns

    for loc in splits.test["location"].unique():
        tr = splits.train.loc[splits.train["location"] == loc, "time"]
        va = splits.val.loc[splits.val["location"] == loc, "time"]
        te = splits.test.loc[splits.test["location"] == loc, "time"]
        assert tr.max() < va.min() < va.max() < te.min(), loc
        span_days = (te.max() - te.min()) / pd.Timedelta(days=1)
        assert abs(span_days - SPLIT_DAYS) < 1.0, (loc, span_days)
    print("ok  split: train < val < test per location, no overlap, ~14d test window")


def check_registry_roundtrip(tmp: Path) -> None:
    registry.MODELS_DIR = tmp  # redirect the registry for the test

    X = np.arange(20).reshape(-1, 2).astype(float)
    y = X.sum(axis=1)
    worse = LinearRegression().fit(X, y)
    better = LinearRegression().fit(X, y + 0.0)
    feats = ["f0", "f1"]

    v1 = registry.register_model(
        "demo", worse, {"val": {"rmse": 3.0}, "test": {"rmse": 5.0}}, feats
    )
    v2 = registry.register_model(
        "demo", better, {"val": {"rmse": 2.0}, "test": {"rmse": 1.0}}, feats
    )
    assert (v1, v2) == (1, 2)

    model, meta = registry.load_best_model("demo")
    assert meta["version"] == 2, meta["version"]
    assert meta["metrics"]["test"]["rmse"] == 1.0
    assert meta["feature_list"] == feats
    np.testing.assert_allclose(model.predict(X), better.predict(X))
    print("ok  registry: round-trips model + metadata, load_best picks lowest test RMSE")


def main() -> int:
    check_target_no_leakage()
    check_horizon_target_no_leakage()
    check_split_non_default_horizon()
    check_split_no_overlap()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        check_registry_roundtrip(Path(tmp))

    print("\nall training-pipeline smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
