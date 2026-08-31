"""Shared regression metrics for the training pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE / MAE / R2 as plain floats. The old ``accuracy`` metric is gone."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
