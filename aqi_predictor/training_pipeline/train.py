"""Train and compare next-hour US AQI regressors.

Models: Ridge (with a train-fit StandardScaler), RandomForest, XGBoost. Each is
evaluated on the validation and test splits with RMSE / MAE / R2. The best model
by test RMSE is written to the local model registry.

Run:  python -m aqi_predictor.training_pipeline.train
"""

from __future__ import annotations

import json

import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from aqi_predictor.config import MODELS_DIR
from aqi_predictor.training_pipeline import registry
from aqi_predictor.training_pipeline.dataset import Splits, split_dataset
from aqi_predictor.training_pipeline.metrics import regression_metrics

MODEL_NAME = "us_aqi_next"
COMPARISON_PATH = MODELS_DIR / "training_comparison.json"

RANDOM_STATE = 42


def build_models() -> dict[str, object]:
    """Fresh, unfitted estimators. Ridge scales inside its pipeline (train-fit)."""
    return {
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return regression_metrics(y, model.predict(X))


def compute_shap_importance(model, X_test: pd.DataFrame) -> dict[str, float] | None:
    """Mean |SHAP value| per feature for tree-based ``model``, else ``None``.

    Ridge is fit inside a ``Pipeline`` so it never matches either isinstance
    check here. XGBRegressor/RandomForestRegressor are the only estimators
    :func:`shap.TreeExplainer` is used for - no background sample needed.
    """
    if not isinstance(model, (RandomForestRegressor, XGBRegressor)):
        return None
    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    mean_abs = pd.DataFrame(shap_values, columns=X_test.columns).abs().mean()
    return {col: float(val) for col, val in mean_abs.items()}


def train_all(splits: Splits) -> dict[str, dict]:
    X_train, y_train = splits.xy("train")
    X_val, y_val = splits.xy("val")
    X_test, y_test = splits.xy("test")

    results: dict[str, dict] = {}
    for name, model in build_models().items():
        print(f"[train] fitting {name} on {len(X_train)} rows ...", flush=True)
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "metrics": {
                "val": evaluate(model, X_val, y_val),
                "test": evaluate(model, X_test, y_test),
            },
        }
    return results


def _format_table(results: dict[str, dict]) -> str:
    header = f"{'model':<16}{'split':<7}{'RMSE':>10}{'MAE':>10}{'R2':>10}"
    lines = [header, "-" * len(header)]
    for name, res in results.items():
        for split in ("val", "test"):
            m = res["metrics"][split]
            lines.append(
                f"{name:<16}{split:<7}{m['rmse']:>10.3f}{m['mae']:>10.3f}{m['r2']:>10.4f}"
            )
    return "\n".join(lines)


def main() -> int:
    splits = split_dataset()
    print(
        f"[train] rows  train={len(splits.train)}  val={len(splits.val)}  "
        f"test={len(splits.test)}  features={len(splits.feature_columns)}"
    )

    results = train_all(splits)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON - next-hour us_aqi")
    print("=" * 60)
    print(_format_table(results))
    print("=" * 60)

    best_name = min(results, key=lambda n: results[n]["metrics"]["test"]["rmse"])
    best = results[best_name]
    print(f"\nbest by test RMSE: {best_name} "
          f"(RMSE={best['metrics']['test']['rmse']:.3f})")

    X_test, _y_test = splits.xy("test")
    shap_importance = compute_shap_importance(best["model"], X_test)

    version = registry.register_model(
        MODEL_NAME,
        best["model"],
        metrics={**best["metrics"], "algorithm": best_name, "selected_by": "test_rmse"},
        feature_list=splits.feature_columns,
        shap_importance=shap_importance,
    )
    print(f"registered {MODEL_NAME} v{version} in the Hopsworks model registry")

    comparison = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "best": best_name,
        "registered_version": version,
        "results": {
            name: res["metrics"] for name, res in results.items()
        },
    }
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"comparison saved -> {COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
