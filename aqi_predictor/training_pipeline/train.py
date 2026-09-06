"""Train and compare next-hour US AQI regressors.

Models: Ridge (with a train-fit StandardScaler), RandomForest, XGBoost. Each is
evaluated on the validation and test splits with RMSE / MAE / R2. The best model
by test RMSE is written to the local model registry.

Run:  python -m aqi_predictor.training_pipeline.train
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from aqi_predictor.config import MODELS_DIR
from aqi_predictor.training_pipeline import registry
from aqi_predictor.training_pipeline.dataset import Splits, split_dataset
from aqi_predictor.training_pipeline.lstm_model import AQI_LSTM, train_lstm
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


def _compute_lstm_shap_importance(lstm_context: dict) -> dict[str, float]:
    """Global mean |SHAP| per feature for the LSTM via ``shap.GradientExplainer``.

    ``shap.DeepExplainer`` was tried first (per the original plan) but its
    PyTorch backend has no attribution rule registered for ``nn.LSTM``
    ("unrecognized nn.Module: LSTM"), and the resulting attributions failed
    shap's own additivity check by ~40x the tolerance - not a rounding error,
    a genuinely unsupported op. ``GradientExplainer`` computes expected
    gradients via plain autograd with no per-layer op registry, so it works on
    any differentiable architecture including LSTMs, with no additivity
    check to fail. Still an approximation (as ``DeepExplainer`` would have
    been), just one that doesn't require an unsupported op.

    Returns SHAP values shaped ``(n_samples, seq_len, n_features)`` (one
    attribution per input timestep). Summing absolute values across the time
    axis first, then averaging over samples, collapses that to the same
    non-negative ``{feature: mean_abs_shap}`` shape ``compute_shap_importance``
    already returns for the tree models - no dashboard change needed.
    """
    model = lstm_context["model"]
    background = torch.tensor(lstm_context["background_scaled"], dtype=torch.float32)
    test_sequences = torch.tensor(
        lstm_context["test_sequences_scaled"], dtype=torch.float32
    )

    model.eval()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_sequences)
    if isinstance(shap_values, list):  # some shap versions wrap single-output in a list
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4:  # trailing output-dim some shap versions add
        shap_values = shap_values[..., 0]

    per_sample_importance = np.abs(shap_values).sum(axis=1)  # (n_samples, n_features)
    mean_abs = per_sample_importance.mean(axis=0)  # (n_features,)
    return {
        col: float(val)
        for col, val in zip(lstm_context["feature_columns"], mean_abs, strict=True)
    }


def compute_shap_importance(
    model, X_test: pd.DataFrame | None = None, lstm_context: dict | None = None
) -> dict[str, float] | None:
    """Mean |SHAP value| per feature for tree-based/LSTM ``model``, else ``None``.

    Ridge is fit inside a ``Pipeline`` so it never matches either isinstance
    check here. XGBRegressor/RandomForestRegressor use ``shap.TreeExplainer``
    (no background sample needed) over the flat ``X_test`` rows; ``AQI_LSTM``
    uses ``shap.GradientExplainer`` over ``lstm_context``'s scaled test
    sequences and background sample (see
    :func:`_compute_lstm_shap_importance` for why not ``DeepExplainer``).
    """
    if isinstance(model, (RandomForestRegressor, XGBRegressor)):
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
        mean_abs = pd.DataFrame(shap_values, columns=X_test.columns).abs().mean()
        return {col: float(val) for col, val in mean_abs.items()}
    if isinstance(model, AQI_LSTM):
        return _compute_lstm_shap_importance({**lstm_context, "model": model})
    return None


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

    print(f"[train] fitting lstm on {len(X_train)} rows ...", flush=True)
    results["lstm"] = train_lstm(splits)
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
    shap_importance = compute_shap_importance(
        best["model"], X_test, lstm_context=best.get("lstm_context")
    )

    version = registry.register_model(
        MODEL_NAME,
        best["model"],
        metrics={**best["metrics"], "algorithm": best_name, "selected_by": "test_rmse"},
        feature_list=splits.feature_columns,
        shap_importance=shap_importance,
        extra_artifacts=best.get("extra_artifacts"),
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
