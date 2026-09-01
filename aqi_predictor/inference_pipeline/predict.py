"""Live multi-horizon US AQI inference.

Pulls current conditions + a real weather *forecast* from Open-Meteo, then runs
each registered horizon model (``us_aqi_next`` for +1h, ``us_aqi_h{24,48,72}``
for the rest) to predict future US AQI.

How this differs from training: in :mod:`aqi_predictor.training_pipeline.dataset`
the ``<var>_target`` columns (weather at target time) are filled with the
*recorded* historical value - a stand-in for "a forecast would have been
available". Here they are filled with an actual forward-looking forecast from the
Open-Meteo forecast endpoint, which is what a deployed system would use.

Run:  python -m aqi_predictor.inference_pipeline.predict --location karachi
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from aqi_predictor.feature_pipeline import fetch
from aqi_predictor.feature_pipeline.features import build_features
from aqi_predictor.training_pipeline import registry

HORIZONS = (1, 24, 48, 72)
_TARGET_SUFFIX = "_target"
_HISTORY_DAYS = 4
RECENT_HOURS = 48  # how much observed history forecast() returns under "recent"


def model_name(horizon_hours: int) -> str:
    """Registry name for a horizon: ``us_aqi_next`` (h=1) or ``us_aqi_h<h>``."""
    return "us_aqi_next" if horizon_hours == 1 else f"us_aqi_h{horizon_hours}"


def _current_and_history(location_name: str) -> tuple[pd.Series, pd.DataFrame]:
    """The most recent fully-featured row at or before the current wall-clock
    hour, plus the observed history it came from (rows ``time <= now``).

    Built from the last ``_HISTORY_DAYS`` days of history run through
    ``features.build_features`` (the same engineering the feature store uses),
    then filtered to ``time <= now``. Open-Meteo's air-quality endpoint forecasts
    the remainder of the current UTC day, so the unfiltered last row can be up to
    ~23h in the future; the filter keeps only real recent hours (same guard
    ``fetch.latest_hour`` uses).
    """
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=_HISTORY_DAYS)).date().isoformat()
    end = now.date().isoformat()

    history = fetch.historical(location_name, start, end)
    featured, _reports = build_features(history)
    if featured.empty:
        raise RuntimeError(
            f"could not build a feature row for {location_name!r} from "
            f"{_HISTORY_DAYS} days of history (build_features returned nothing)"
        )

    current_hour = now.floor("h")
    observed = featured[featured["time"] <= current_hour].reset_index(drop=True)
    if observed.empty:
        raise RuntimeError(
            f"no fully-featured row for {location_name!r} at or before "
            f"{current_hour} (fetched {len(featured)} rows spanning "
            f"{featured['time'].min()} .. {featured['time'].max()})"
        )
    return observed.iloc[-1], observed


def build_input_row(
    now_row: pd.Series,
    feature_list: list[str],
    forecast_weather: pd.DataFrame,
    target_time: pd.Timestamp,
) -> pd.DataFrame:
    """One-row DataFrame with **exactly** ``feature_list`` (order preserved).

    ``<var>_target`` columns take the forecast value at ``target_time``; every
    other column is copied from ``now_row``. Any column that cannot be sourced
    raises - the model input never silently mismatches its ``feature_list``.
    """
    fc = forecast_weather.set_index("time")
    if target_time not in fc.index:
        raise RuntimeError(
            f"forecast weather does not cover target time {target_time} "
            f"(available {fc.index.min()} .. {fc.index.max()})"
        )

    values: dict[str, object] = {}
    missing: list[str] = []
    for col in feature_list:
        if col.endswith(_TARGET_SUFFIX):
            base = col[: -len(_TARGET_SUFFIX)]
            if base in fc.columns:
                values[col] = fc.at[target_time, base]
            else:
                missing.append(col)
        elif col in now_row.index:
            values[col] = now_row[col]
        else:
            missing.append(col)

    if missing:
        raise RuntimeError(
            f"cannot build model input: {len(missing)} feature(s) not available "
            f"from the current row or the weather forecast: {missing}"
        )

    row = pd.DataFrame([[values[c] for c in feature_list]], columns=list(feature_list))
    if list(row.columns) != list(feature_list):
        raise RuntimeError("model input columns do not match feature_list")
    return row.astype("float64")


def forecast(location_name: str) -> dict:
    """Predict US AQI at every horizon in :data:`HORIZONS` for ``location_name``."""
    now_row, observed = _current_and_history(location_name)
    now_time = pd.Timestamp(now_row["time"])
    weather = fetch.forecast_ahead(location_name, hours_ahead=max(HORIZONS))

    forecasts = []
    for horizon in HORIZONS:
        model, meta = registry.load_best_model(model_name(horizon))
        target_time = now_time + pd.Timedelta(hours=horizon)
        x = build_input_row(now_row, meta["feature_list"], weather, target_time)
        predicted = float(model.predict(x)[0])
        forecasts.append(
            {
                "horizon_hours": horizon,
                "target_time": target_time.isoformat(),
                "predicted_us_aqi": round(predicted, 1),
                "model_name": meta["name"],
                "model_version": meta["version"],
            }
        )

    recent_cut = now_time - pd.Timedelta(hours=RECENT_HOURS)
    recent = [
        {"time": t.isoformat(), "us_aqi": round(float(v), 1)}
        for t, v in zip(observed["time"], observed["us_aqi"])
        if t >= recent_cut
    ]

    return {
        "location": location_name,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "current": {
            "time": now_time.isoformat(),
            "us_aqi": round(float(now_row["us_aqi"]), 1),
        },
        "recent": recent,
        "forecasts": forecasts,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pretty(result: dict) -> str:
    cur = result["current"]
    lines = [
        f"US AQI forecast - {result['location']}",
        f"  generated_at : {result['generated_at']}",
        f"  current      : us_aqi {cur['us_aqi']:.0f}  @ {cur['time']}",
        "",
        f"  {'horizon':>7}  {'target time (UTC)':<25}  {'us_aqi':>7}  model",
        f"  {'-' * 7}  {'-' * 25}  {'-' * 7}  {'-' * 22}",
    ]
    for f in result["forecasts"]:
        lines.append(
            f"  {'+' + str(f['horizon_hours']) + 'h':>7}  "
            f"{f['target_time']:<25}  {f['predicted_us_aqi']:>7.1f}  "
            f"{f['model_name']} v{f['model_version']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live multi-horizon US AQI forecast")
    parser.add_argument("--location", default="karachi", help="location name")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args()

    result = forecast(args.location)
    print(json.dumps(result, indent=2) if args.json else _pretty(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
