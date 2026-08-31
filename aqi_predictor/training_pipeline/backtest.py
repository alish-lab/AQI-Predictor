"""Recursive multi-horizon backtest over the test period.

For every hour in the test window (that has room for a full 72h horizon) each
model is rolled forward one hour at a time to +72h. At each step the synthetic
feature row is rebuilt:

* ``us_aqi`` and everything derived from it (lags, rolling stats, change-rate)
  use the model's own predicted trajectory spliced onto real history;
* weather columns take their **real** historical values (a stand-in for a real
  weather forecast being available at inference time);
* time columns are recomputed from the timestamp;
* everything else - notably ``pm2_5`` / ``pm10`` and their lag/rolling features,
  plus the other pollutants - is **frozen at its value at the forecast origin**.
  The model only predicts ``us_aqi``, so this is a deliberate, documented
  simplification (see PROGRESS.md).

Reported: RMSE / MAE / R2 at +24h, +48h and +72h for each model.

Run:  python -m aqi_predictor.training_pipeline.backtest
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqi_predictor.config import MODELS_DIR, WEATHER_VARS
from aqi_predictor.feature_pipeline import store
from aqi_predictor.training_pipeline.dataset import split_dataset
from aqi_predictor.training_pipeline.metrics import regression_metrics
from aqi_predictor.training_pipeline.train import train_all

HORIZONS = (24, 48, 72)
MAX_H = max(HORIZONS)
_LOOKBACK = 24  # hours of real history needed for the 24h windows

_TIME_COLUMNS = {
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
}
_RESULT_PATH = MODELS_DIR / "backtest_report.json"


# --------------------------------------------------------------------------- #
# Synthetic feature construction
# --------------------------------------------------------------------------- #
def _time_feature(col: str, ts: pd.DatetimeIndex) -> np.ndarray:
    hour = np.asarray(ts.hour, dtype=float)
    dow = np.asarray(ts.dayofweek, dtype=float)
    month = np.asarray(ts.month, dtype=float)
    if col == "hour":
        return hour
    if col == "day_of_week":
        return dow
    if col == "month":
        return month
    if col == "is_weekend":
        return (dow >= 5).astype(float)
    if col == "hour_sin":
        return np.sin(2 * np.pi * hour / 24)
    if col == "hour_cos":
        return np.cos(2 * np.pi * hour / 24)
    if col == "month_sin":
        return np.sin(2 * np.pi * (month - 1) / 12)
    if col == "month_cos":
        return np.cos(2 * np.pi * (month - 1) / 12)
    raise KeyError(col)


def _us_aqi_feature(col: str, u: np.ndarray, p: int) -> np.ndarray:
    """Column derived from the evolving us_aqi buffer ``u`` at current index ``p``."""
    if col == "us_aqi":
        return u[:, p]
    if col == "us_aqi_lag_1h":
        return u[:, p - 1]
    if col == "us_aqi_lag_2h":
        return u[:, p - 2]
    if col == "us_aqi_lag_3h":
        return u[:, p - 3]
    if col == "us_aqi_change_1h":
        return u[:, p] - u[:, p - 1]
    if col == "us_aqi_change_24h":
        return u[:, p] - u[:, p - 24]
    if col.startswith("us_aqi_roll_"):
        stat, win = col[len("us_aqi_roll_") :].split("_")  # e.g. "mean", "3h"
        w = int(win.rstrip("h"))
        window = u[:, p - w + 1 : p + 1]
        return window.mean(axis=1) if stat == "mean" else window.std(axis=1, ddof=1)
    raise KeyError(col)


def _categorise(feature_cols: list[str]) -> dict[str, list[str]]:
    us_aqi, weather, time_cols, frozen = [], [], [], []
    for c in feature_cols:
        if c == "us_aqi" or c.startswith("us_aqi_"):
            us_aqi.append(c)
        elif c in WEATHER_VARS:
            weather.append(c)
        elif c in _TIME_COLUMNS:
            time_cols.append(c)
        else:
            frozen.append(c)
    return {"us_aqi": us_aqi, "weather": weather, "time": time_cols, "frozen": frozen}


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def _location_backtest(
    models: dict[str, object],
    feature_cols: list[str],
    history: pd.DataFrame,
    test_times: pd.Series,
) -> tuple[dict[str, dict[int, dict[str, float]]], int]:
    """Recursive backtest for one location. ``history`` is the full gap-free
    hourly feature frame for that location, time-sorted."""
    history = history.sort_values("time").reset_index(drop=True)
    if not (history["time"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise RuntimeError("backtest needs a gap-free hourly history")

    time_index = pd.DatetimeIndex(history["time"])
    pos_of = {t: i for i, t in enumerate(time_index)}
    last_pos = len(history) - 1

    origins = [
        t for t in test_times
        if t in pos_of
        and pos_of[t] - _LOOKBACK >= 0
        and pos_of[t] + MAX_H <= last_pos
    ]
    if not origins:
        raise RuntimeError("no test origins with a full 72h horizon of real data")
    origin_pos = np.array([pos_of[t] for t in origins])
    n = len(origins)

    us_aqi_all = history["us_aqi"].to_numpy(dtype=float)
    cats = _categorise(feature_cols)

    # real future values, indexed [origin, step]
    weather_future = {
        c: history[c].to_numpy(dtype=float) for c in cats["weather"]
    }
    frozen_at_origin = history.iloc[origin_pos][cats["frozen"]].to_numpy(dtype=float)
    frozen_col_pos = {c: i for i, c in enumerate(cats["frozen"])}

    # actual us_aqi at each horizon
    actuals = {h: us_aqi_all[origin_pos + h] for h in HORIZONS}

    results: dict[str, dict[int, dict[str, float]]] = {}
    for name, model in models.items():
        # u[:, j] == us_aqi at (origin - _LOOKBACK + j); predictions fill j > _LOOKBACK
        u = np.empty((n, _LOOKBACK + MAX_H + 1))
        for j in range(_LOOKBACK + 1):
            u[:, j] = us_aqi_all[origin_pos - _LOOKBACK + j]

        preds: dict[int, np.ndarray] = {}
        for k in range(MAX_H):
            p = _LOOKBACK + k
            step_pos = origin_pos + k
            step_ts = time_index[step_pos]

            X = np.empty((n, len(feature_cols)))
            for ci, col in enumerate(feature_cols):
                if col in cats["us_aqi"]:
                    X[:, ci] = _us_aqi_feature(col, u, p)
                elif col in weather_future:
                    X[:, ci] = weather_future[col][step_pos]
                elif col in _TIME_COLUMNS:
                    X[:, ci] = _time_feature(col, step_ts)
                else:
                    X[:, ci] = frozen_at_origin[:, frozen_col_pos[col]]

            u[:, p + 1] = model.predict(pd.DataFrame(X, columns=feature_cols))
            if (k + 1) in HORIZONS:
                preds[k + 1] = u[:, p + 1]

        results[name] = {
            h: regression_metrics(actuals[h], preds[h]) for h in HORIZONS
        }
    return results, n


def run_backtest(models: dict[str, object] | None = None) -> dict:
    splits = split_dataset()
    feature_cols = splits.feature_columns

    if models is None:
        print("[backtest] training models ...", flush=True)
        trained = train_all(splits)
        models = {name: res["model"] for name, res in trained.items()}

    full = store.get_feature_view("2000-01-01", "2100-01-01")

    per_location: dict[str, dict] = {}
    for location, hist in full.groupby("location"):
        test_times = splits.test.loc[splits.test["location"] == location, "time"]
        if test_times.empty:
            continue
        res, n_origins = _location_backtest(models, feature_cols, hist, test_times)
        per_location[location] = {"n_origins": n_origins, "metrics": res}

    report = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "horizons_h": list(HORIZONS),
        "per_location": per_location,
    }
    _RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: dict) -> None:
    for location, block in report["per_location"].items():
        print("\n" + "=" * 66)
        print(f"RECURSIVE BACKTEST - {location}  ({block['n_origins']} forecast origins)")
        print("=" * 66)
        header = f"{'model':<16}{'horizon':>9}{'RMSE':>11}{'MAE':>11}{'R2':>10}"
        print(header)
        print("-" * len(header))
        for name, by_h in block["metrics"].items():
            for h in report["horizons_h"]:
                m = by_h[h] if h in by_h else by_h[str(h)]
                print(
                    f"{name:<16}{str(h) + 'h':>9}{m['rmse']:>11.3f}"
                    f"{m['mae']:>11.3f}{m['r2']:>10.4f}"
                )
        print("=" * 66)


def main() -> int:
    report = run_backtest()
    _print_report(report)
    print(f"\nbacktest report saved -> {_RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
