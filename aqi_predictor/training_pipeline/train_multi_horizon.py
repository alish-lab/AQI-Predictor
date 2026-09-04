"""Train direct multi-horizon US AQI models (+24h / +48h / +72h).

Walking the 1-hour model forward recursively compounded error badly - R2 went
negative past +24h (see PROGRESS.md, "Phase 2b"). This trains a separate model
per horizon that predicts the future value directly from *current* known
features, so there is no step-by-step error accumulation and no need to freeze
pm2.5 / pm10.

For each horizon: build the dataset at that horizon (features include
``<var>_target`` = weather at target time, added by ``dataset.py``), train
Ridge / RandomForest / XGBoost (same estimators as ``train.py``), evaluate on
val + test, and register the best-by-test-RMSE model as ``us_aqi_h24`` /
``us_aqi_h48`` / ``us_aqi_h72``.

Run:  python -m aqi_predictor.training_pipeline.train_multi_horizon
"""

from __future__ import annotations

import json

import pandas as pd

from aqi_predictor.config import MODELS_DIR
from aqi_predictor.training_pipeline import registry
from aqi_predictor.training_pipeline.dataset import split_dataset, target_name
from aqi_predictor.training_pipeline.train import (
    _format_table,
    compute_shap_importance,
    train_all,
)

HORIZONS = (24, 48, 72)
COMPARISON_PATH = MODELS_DIR / "training_comparison_multi_horizon.json"


def train_and_register(horizon_hours: int) -> dict:
    """Train all models at one horizon, register the best, return a summary dict."""
    model_name = target_name(horizon_hours)  # us_aqi_h24 / h48 / h72
    splits = split_dataset(horizon_hours=horizon_hours)
    print(
        f"\n[+{horizon_hours}h] rows  train={len(splits.train)}  "
        f"val={len(splits.val)}  test={len(splits.test)}  "
        f"features={len(splits.feature_columns)}  target={splits.target}"
    )

    results = train_all(splits)

    print("\n" + "=" * 60)
    print(f"MODEL COMPARISON - us_aqi +{horizon_hours}h")
    print("=" * 60)
    print(_format_table(results))
    print("=" * 60)

    best_name = min(results, key=lambda n: results[n]["metrics"]["test"]["rmse"])
    best = results[best_name]
    print(
        f"best by test RMSE: {best_name} "
        f"(RMSE={best['metrics']['test']['rmse']:.3f}, "
        f"R2={best['metrics']['test']['r2']:.4f})"
    )

    X_test, _y_test = splits.xy("test")
    shap_importance = compute_shap_importance(best["model"], X_test)

    version = registry.register_model(
        model_name,
        best["model"],
        metrics={
            **best["metrics"],
            "algorithm": best_name,
            "selected_by": "test_rmse",
            "horizon_hours": horizon_hours,
        },
        feature_list=splits.feature_columns,
        shap_importance=shap_importance,
    )
    print(f"registered {model_name} v{version}")

    return {
        "horizon_hours": horizon_hours,
        "registered_as": model_name,
        "registered_version": version,
        "best": best_name,
        "rows": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "results": {name: res["metrics"] for name, res in results.items()},
    }


def main() -> int:
    combined = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "horizons": {},
    }
    for horizon in HORIZONS:
        summary = train_and_register(horizon)
        combined["horizons"][str(horizon)] = summary

    COMPARISON_PATH.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\ncombined comparison saved -> {COMPARISON_PATH}")

    print("\n" + "=" * 60)
    print("DIRECT MULTI-HORIZON SUMMARY (best model per horizon, test split)")
    print("=" * 60)
    print(f"{'horizon':<10}{'model':<16}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    print("-" * 56)
    for horizon in HORIZONS:
        s = combined["horizons"][str(horizon)]
        m = s["results"][s["best"]]["test"]
        print(
            f"+{horizon}h{'':<6}{s['best']:<16}"
            f"{m['rmse']:>10.3f}{m['mae']:>10.3f}{m['r2']:>10.4f}"
        )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
