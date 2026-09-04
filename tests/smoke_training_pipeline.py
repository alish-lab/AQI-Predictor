"""Offline smoke checks for the training pipeline (no network, no pytest).

Run:  python tests/smoke_training_pipeline.py

Covers the Phase 2 / 2b definition-of-done invariants:
* the ``us_aqi_next`` target is a genuine future value (no leakage) and each
  location's last row is dropped,
* the horizon-parameterized target (``horizon_hours=24``) is the real value 24h
  later, and ``split_dataset`` works at a non-default horizon,
* the ``<var>_target`` weather-at-target-time features are the real weather
  values ``horizon`` hours later (no leakage),
* the time-ordered train / val / test split has no overlap,
* the model registry round-trips a model + full nested metadata and picks "best"
  by test RMSE (against an in-memory fake Hopsworks registry - no network).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from aqi_predictor.training_pipeline import dataset, registry
from aqi_predictor.training_pipeline.train import compute_shap_importance


# --------------------------------------------------------------------------- #
# In-memory stand-in for the Hopsworks Model Registry (no network in the tests).
# Keyed by (name, version); artifacts are copied to a temp dir per version so
# registry.load_model / load_best_model still exercise the real download + read
# + joblib.load path.
# --------------------------------------------------------------------------- #
class _FakeModel:
    def __init__(self, mr: "_FakeModelRegistry", name: str, metrics: dict) -> None:
        self._mr = mr
        self.name = name
        self.training_metrics = dict(metrics)
        self.version: int | None = None
        self._dir: Path | None = None

    def save(self, src_dir: str) -> "_FakeModel":
        self.version = self._mr._next_version(self.name)
        dst = Path(tempfile.mkdtemp(prefix=f"fakemr_{self.name}_v{self.version}_"))
        for item in Path(src_dir).iterdir():
            shutil.copy2(item, dst / item.name)
        self._dir = dst
        self._mr._store[(self.name, self.version)] = self
        return self

    def download(self) -> str:
        return str(self._dir)


class _FakeModelRegistryNS:
    def __init__(self, mr: "_FakeModelRegistry") -> None:
        self._mr = mr

    def create_model(self, name: str, metrics: dict, description: str = "") -> _FakeModel:
        return _FakeModel(self._mr, name, metrics)


class _FakeModelRegistry:
    def __init__(self) -> None:
        self._store: dict[tuple[str, int], _FakeModel] = {}
        self.python = _FakeModelRegistryNS(self)

    def _next_version(self, name: str) -> int:
        existing = [v for (n, v) in self._store if n == name]
        return (max(existing) + 1) if existing else 1

    def get_models(self, name: str) -> list[_FakeModel]:
        return [m for (n, _v), m in self._store.items() if n == name]

    def get_model(self, name: str, version: int) -> _FakeModel | None:
        return self._store.get((name, version))


class _FakeProject:
    def __init__(self, mr: _FakeModelRegistry) -> None:
        self._mr = mr

    def get_model_registry(self) -> _FakeModelRegistry:
        return self._mr
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


def check_weather_target_no_leakage() -> None:
    horizon = 48
    wcol = "temperature_2m"          # present in _synthetic()
    tcol = f"{wcol}_target"

    raw = _synthetic()
    frame = build_training_frame(raw, horizon_hours=horizon)
    assert tcol in frame.columns, tcol
    assert tcol in split_dataset(frame, horizon_hours=horizon).feature_columns

    for loc, g in frame.groupby("location"):
        g = g.sort_values("time").reset_index(drop=True)
        # {wcol}_target at time t must equal the real weather value at t + horizon
        merged = g.merge(
            g[["time", wcol]].rename(columns={"time": "t_future", wcol: "w_future"}),
            left_on=g["time"] + pd.Timedelta(hours=horizon),
            right_on="t_future",
            how="inner",
        )
        assert len(merged) > 0
        assert np.allclose(merged[tcol], merged["w_future"]), loc
    assert not frame[tcol].isna().any()
    print(
        "ok  weather target: temperature_2m_target is the real value 48h later"
    )


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


def check_registry_roundtrip() -> None:
    project = _FakeProject(_FakeModelRegistry())
    registry._project = lambda: project  # swap the Hopsworks handle for a fake

    X = np.arange(20).reshape(-1, 2).astype(float)
    y = X.sum(axis=1)
    worse = LinearRegression().fit(X, y)
    better = LinearRegression().fit(X, y + 0.0)
    feats = ["f0", "f1"]

    v1 = registry.register_model(
        "demo", worse,
        {"val": {"rmse": 3.0, "mae": 2.0, "r2": 0.5},
         "test": {"rmse": 5.0, "mae": 4.0, "r2": 0.3},
         "algorithm": "linreg"},
        feats,
        # no shap_importance -> default None, must not break registration
    )
    shap_importance = {"f0": 0.7, "f1": 0.3}
    v2 = registry.register_model(
        "demo", better,
        {"val": {"rmse": 2.0, "mae": 1.5, "r2": 0.7},
         "test": {"rmse": 1.0, "mae": 0.8, "r2": 0.9},
         "algorithm": "linreg"},
        feats,
        shap_importance=shap_importance,
    )
    assert (v1, v2) == (1, 2)

    assert [m["version"] for m in registry.list_versions("demo")] == [1, 2]

    model, meta = registry.load_best_model("demo")
    assert meta["version"] == 2, meta["version"]
    assert meta["metrics"]["test"]["rmse"] == 1.0
    assert meta["metrics"]["algorithm"] == "linreg"  # nested dict preserved
    assert meta["feature_list"] == feats
    assert meta["shap_importance"] == shap_importance  # round-trips through the registry
    np.testing.assert_allclose(model.predict(X), better.predict(X))

    m1, meta1 = registry.load_model("demo", 1)
    assert meta1["version"] == 1
    assert meta1["shap_importance"] is None  # default stays None when omitted
    np.testing.assert_allclose(m1.predict(X), worse.predict(X))

    # v3 ties v2 on test RMSE -> tie must break toward the higher version, not
    # whichever the (fake) API happened to list first.
    tied = LinearRegression().fit(X, y + 0.0)
    v3 = registry.register_model(
        "demo", tied,
        {"val": {"rmse": 2.0, "mae": 1.5, "r2": 0.7},
         "test": {"rmse": 1.0, "mae": 0.8, "r2": 0.9},  # tied with v2
         "algorithm": "linreg"},
        feats,
    )
    assert v3 == 3
    _model3, meta3 = registry.load_best_model("demo")
    assert meta3["version"] == 3, meta3["version"]  # higher version wins the tie

    # v4 is a near-tie with v3 (RMSE differs only past the 6th decimal, like
    # RandomForest's run-to-run float noise) -> must still count as a tie.
    near_tied = LinearRegression().fit(X, y + 0.0)
    v4 = registry.register_model(
        "demo", near_tied,
        {"val": {"rmse": 2.0, "mae": 1.5, "r2": 0.7},
         "test": {"rmse": 1.0 + 5e-9, "mae": 0.8, "r2": 0.9},  # near-tied with v3
         "algorithm": "linreg"},
        feats,
    )
    assert v4 == 4
    _model4, meta4 = registry.load_best_model("demo")
    assert meta4["version"] == 4, meta4["version"]  # rounded tie -> higher version wins

    print("ok  registry: round-trips model + metadata (incl. shap_importance), "
          "load_best picks lowest test RMSE (ties -> higher version)")


def check_shap_importance() -> None:
    X = pd.DataFrame(
        np.arange(40).reshape(-1, 2).astype(float), columns=["f0", "f1"]
    )
    y = X["f0"] + 2 * X["f1"]

    tree_model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    importance = compute_shap_importance(tree_model, X)
    assert importance is not None
    assert set(importance) == {"f0", "f1"}
    assert all(isinstance(v, float) and v >= 0 for v in importance.values())

    linear_model = LinearRegression().fit(X, y)
    assert compute_shap_importance(linear_model, X) is None
    print("ok  shap: tree model -> mean |SHAP| per feature, non-tree model -> None")


def main() -> int:
    check_target_no_leakage()
    check_horizon_target_no_leakage()
    check_weather_target_no_leakage()
    check_split_non_default_horizon()
    check_split_no_overlap()
    check_registry_roundtrip()
    check_shap_importance()

    print("\nall training-pipeline smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
