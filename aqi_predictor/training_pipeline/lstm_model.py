"""LSTM sequence model for US AQI forecasting.

Where the tree/linear models in :mod:`train.py` see one engineered-feature row
per prediction, this model sees the trailing :data:`SEQ_LEN`-hour trajectory of
the same engineered columns (reused as-is, not raw un-engineered inputs) and
predicts the same per-horizon ``us_aqi`` target. It is registered under the
same model names and competes on test RMSE via the existing
``registry.load_best_model`` tie-break - there is no special-casing to make it
win or lose.

:func:`build_sequences` slides a ``SEQ_LEN``-hour window over one split
(train/val/test) at a time, per location, so windows never cross a location or
a split boundary; a window is only formed over a run of truly-consecutive
hours (checked from the ``time`` column), so a gap the feature pipeline could
not interpolate silently drops that window instead of splicing non-adjacent
hours together.

:func:`train_lstm` fits a train-only ``StandardScaler`` (the tree models need
no scaling; this one does), trains :class:`AQI_LSTM` with early stopping on
validation RMSE, and returns everything :mod:`train.py` needs to register it
via the registry's ``extra_artifacts`` mechanism and to run
``shap.GradientExplainer`` for the global feature-importance chart
(``DeepExplainer`` has no attribution rule for ``nn.LSTM`` - see
``train.py._compute_lstm_shap_importance``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aqi_predictor.training_pipeline.dataset import Splits
from aqi_predictor.training_pipeline.metrics import regression_metrics

SEQ_LEN = 48
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 8  # epochs without val-RMSE improvement before stopping
BATCH_SIZE = 64
BACKGROUND_SIZE = 64  # random sample of scaled train sequences kept for shap.GradientExplainer
RANDOM_STATE = 42


class AQI_LSTM(nn.Module):
    """1-2 layer LSTM regressor over a ``seq_len``-hour window of features.

    ``seq_len`` is stored as a plain attribute (not a buffer/parameter) so it
    survives ``joblib`` pickling and tells callers - e.g.
    ``predict.py``'s dispatch - how much observed history to feed in.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        seq_len: int = SEQ_LEN,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns shape ``(batch, 1)``, not squeezed to ``(batch,)``.

        shap's PyTorch explainer wrappers index ``outputs.shape[1]`` to detect
        a multi-output model and error (``IndexError: tuple index out of
        range``) on a 1-D output, so the trailing size-1 dimension is kept;
        callers needing a plain vector/scalar (the training loop's metrics,
        ``predict.py``'s single-sequence forecast) squeeze it off after the
        forward pass instead.
        """
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.fc(last)


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding ``seq_len``-hour windows per location: ``(X, y)``.

    ``X`` has shape ``(n_windows, seq_len, len(feature_columns))``; ``y`` has
    shape ``(n_windows,)`` and is ``target`` at the window's last (most
    recent) hour. ``df`` should be a single split (train/val/test) - passing
    one split at a time is what keeps windows from crossing a split boundary.
    A window is skipped if any consecutive pair of its ``seq_len`` hours is
    not exactly 1 hour apart (a gap the feature pipeline left unfilled).
    """
    n_features = len(feature_columns)
    X_parts: list[np.ndarray] = []
    y_parts: list[float] = []

    for _, g in df.sort_values("time").groupby("location", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < seq_len:
            continue
        times = g["time"].to_numpy()
        feats = g[feature_columns].to_numpy(dtype="float64")
        targets = g[target].to_numpy(dtype="float64")
        deltas_h = np.diff(times) / np.timedelta64(1, "h")

        for end in range(seq_len - 1, len(g)):
            start = end - seq_len + 1
            if not np.all(deltas_h[start:end] == 1.0):
                continue
            X_parts.append(feats[start : end + 1])
            y_parts.append(targets[end])

    if not X_parts:
        return np.empty((0, seq_len, n_features)), np.empty((0,))
    return np.stack(X_parts), np.array(y_parts, dtype="float64")


def _fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a ``StandardScaler`` on train-split feature values only."""
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    return scaler


def _scale(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, seq_len, n_features = X.shape
    return scaler.transform(X.reshape(-1, n_features)).reshape(n, seq_len, n_features)


def train_lstm(splits: Splits) -> dict:
    """Train :class:`AQI_LSTM` on ``splits`` and return a ``train_all``-shaped result.

    Returns a dict with ``model``, ``metrics`` (``{"val": {...}, "test": {...}}``,
    same shape ``metrics.regression_metrics`` returns for the tree models),
    ``extra_artifacts`` (for ``registry.register_model``), and ``lstm_context``
    (test sequences + SHAP background, for ``train.py``'s
    ``compute_shap_importance`` to run ``shap.GradientExplainer`` without
    re-deriving them).
    """
    torch.manual_seed(RANDOM_STATE)

    X_train, y_train = build_sequences(splits.train, splits.feature_columns, splits.target)
    X_val, y_val = build_sequences(splits.val, splits.feature_columns, splits.target)
    X_test, y_test = build_sequences(splits.test, splits.feature_columns, splits.target)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError(
            "not enough contiguous hourly rows in one or more splits to build "
            f"a single {SEQ_LEN}h LSTM window (train={len(X_train)}, "
            f"val={len(X_val)}, test={len(X_test)} windows)"
        )

    scaler = _fit_scaler(X_train)
    X_train_s = _scale(X_train, scaler)
    X_val_s = _scale(X_val, scaler)
    X_test_s = _scale(X_test, scaler)

    n_features = X_train_s.shape[2]
    model = AQI_LSTM(n_features=n_features)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_rmse = torch.sqrt(criterion(model(X_val_t), y_val_t)).item()

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"[lstm] early stopping at epoch {epoch} "
                    f"(best val RMSE={best_val_rmse:.3f})",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).numpy().reshape(-1)
        test_pred = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy().reshape(-1)

    metrics = {
        "val": regression_metrics(y_val, val_pred),
        "test": regression_metrics(y_test, test_pred),
    }

    rng = np.random.default_rng(RANDOM_STATE)
    bg_size = min(BACKGROUND_SIZE, len(X_train_s))
    bg_idx = rng.choice(len(X_train_s), size=bg_size, replace=False)
    background_scaled = X_train_s[bg_idx]

    return {
        "model": model,
        "metrics": metrics,
        "extra_artifacts": {
            "scaler.joblib": scaler,
            "shap_background.joblib": background_scaled,
        },
        "lstm_context": {
            "test_sequences_scaled": X_test_s,
            "background_scaled": background_scaled,
            "feature_columns": splits.feature_columns,
        },
    }
