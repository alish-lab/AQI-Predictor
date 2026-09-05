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

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from aqi_predictor.feature_pipeline import fetch
from aqi_predictor.feature_pipeline.features import build_features
from aqi_predictor.training_pipeline import registry
from aqi_predictor.training_pipeline.lstm_model import AQI_LSTM

_TREE_MODEL_TYPES = (RandomForestRegressor, XGBRegressor)
_TOP_N_FEATURES = 5

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


def build_input_sequence(
    window: pd.DataFrame,
    feature_list: list[str],
    forecast_weather: pd.DataFrame,
    horizon_hours: int,
) -> pd.DataFrame:
    """Sequence counterpart to :func:`build_input_row`: one row per hour in
    ``window`` (the trailing ``seq_len`` observed hours, chronological order
    preserved), each with **exactly** ``feature_list`` columns.

    Each row's own ``<var>_target`` uses the forecast value at *that row's*
    ``time + horizon_hours`` - not ``target_time``, which is only the last
    window row's target time. Earlier rows in the window need weather further
    in the past relative to "now" (or, for large horizons, still in the
    future); ``forecast_weather`` must cover the full span, which it does here
    because ``forecast_ahead`` is called with ``past_days=2`` (see
    :mod:`aqi_predictor.feature_pipeline.fetch`).
    """
    fc = forecast_weather.set_index("time")
    out_rows: list[list[float]] = []

    for _, row in window.iterrows():
        row_target_time = pd.Timestamp(row["time"]) + pd.Timedelta(hours=horizon_hours)
        values: dict[str, object] = {}
        missing: list[str] = []
        for col in feature_list:
            if col.endswith(_TARGET_SUFFIX):
                base = col[: -len(_TARGET_SUFFIX)]
                if base in fc.columns and row_target_time in fc.index:
                    values[col] = fc.at[row_target_time, base]
                else:
                    missing.append(col)
            elif col in row.index:
                values[col] = row[col]
            else:
                missing.append(col)

        if missing:
            raise RuntimeError(
                f"cannot build LSTM sequence input at {row['time']} "
                f"(target {row_target_time}): {len(missing)} feature(s) not "
                f"available from the observed row or the weather forecast: {missing}"
            )
        out_rows.append([values[c] for c in feature_list])

    return pd.DataFrame(out_rows, columns=list(feature_list)).astype("float64")


def _lstm_local_shap(
    model: AQI_LSTM,
    x_scaled: torch.Tensor,
    background_scaled: np.ndarray | None,
    feature_columns: list[str],
    raw_last_row: pd.Series,
) -> list[dict] | None:
    """Top ``_TOP_N_FEATURES`` features by |SHAP value| for one LSTM input sequence.

    Uses ``shap.GradientExplainer``, not ``shap.DeepExplainer``: the latter has
    no attribution rule for ``nn.LSTM`` and its output failed shap's own
    additivity check by ~40x tolerance when tried against the real trained
    model (see ``train.py._compute_lstm_shap_importance``).
    ``GradientExplainer`` computes expected gradients via plain autograd, which
    works on any differentiable architecture.

    Returns values shaped ``(1, seq_len, n_features)`` for a single input;
    summing (signed, not absolute) across the time axis collapses that to one
    value per feature while preserving the direction the tree-model path
    already carries (positive = pushes AQI up), so the dashboard's existing
    up/down coloring stays meaningful. ``value`` is the feature's raw value at
    the most recent (last) hour in the window, the closest LSTM analogue to
    the tree path's single-row input value.
    """
    if background_scaled is None:
        return None

    model.eval()
    background = torch.tensor(background_scaled, dtype=torch.float32)
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    per_feature = shap_values[0].sum(axis=0)  # signed sum across seq_len -> (n_features,)
    ranked = sorted(
        zip(feature_columns, per_feature),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:_TOP_N_FEATURES]
    return [
        {
            "feature": feat,
            "shap_value": float(sv),
            "value": float(raw_last_row[feat]) if feat in raw_last_row.index else float("nan"),
        }
        for feat, sv in ranked
    ]


def _local_shap_top_features(model, x: pd.DataFrame) -> list[dict] | None:
    """Top ``_TOP_N_FEATURES`` features by |SHAP value| for one input row.

    ``None`` for non-tree models (Ridge etc.) - only
    ``RandomForestRegressor``/``XGBRegressor`` get a ``shap.TreeExplainer``.
    """
    if not isinstance(model, _TREE_MODEL_TYPES):
        return None
    shap_values = shap.TreeExplainer(model).shap_values(x)[0]
    ranked = sorted(
        zip(x.columns, shap_values, x.iloc[0]),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:_TOP_N_FEATURES]
    return [
        {"feature": feat, "shap_value": float(sv), "value": float(val)}
        for feat, sv, val in ranked
    ]


def forecast(location_name: str) -> dict:
    """Predict US AQI at every horizon in :data:`HORIZONS` for ``location_name``."""
    now_row, observed = _current_and_history(location_name)
    now_time = pd.Timestamp(now_row["time"])
    weather = fetch.forecast_ahead(location_name, hours_ahead=max(HORIZONS))

    forecasts = []
    for horizon in HORIZONS:
        model, meta = registry.load_best_model(model_name(horizon))
        target_time = now_time + pd.Timedelta(hours=horizon)

        if isinstance(model, AQI_LSTM):
            seq_len = model.seq_len
            if len(observed) < seq_len:
                raise RuntimeError(
                    f"only {len(observed)} observed hour(s) available for "
                    f"{location_name!r}, need >= {seq_len} for {meta['name']} "
                    f"v{meta['version']} (an LSTM)"
                )
            window = observed.iloc[-seq_len:].reset_index(drop=True)
            x_seq = build_input_sequence(window, meta["feature_list"], weather, horizon)
            scaler = meta["extra_artifacts"]["scaler.joblib"]
            x_scaled = scaler.transform(x_seq.to_numpy()).reshape(1, seq_len, -1)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                predicted = float(model(x_tensor).squeeze().item())
            top_features = _lstm_local_shap(
                model,
                x_tensor,
                meta["extra_artifacts"].get("shap_background.joblib"),
                meta["feature_list"],
                x_seq.iloc[-1],
            )
        else:
            x = build_input_row(now_row, meta["feature_list"], weather, target_time)
            predicted = float(model.predict(x)[0])
            top_features = _local_shap_top_features(model, x)

        forecasts.append(
            {
                "horizon_hours": horizon,
                "target_time": target_time.isoformat(),
                "predicted_us_aqi": round(predicted, 1),
                "model_name": meta["name"],
                "model_version": meta["version"],
                "top_features": top_features,
                "shap_importance": meta.get("shap_importance"),
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
