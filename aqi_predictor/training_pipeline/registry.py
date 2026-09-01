"""Hopsworks Model Registry integration for the training pipeline.

Public API (unchanged from the earlier local-joblib implementation - every call
site keeps working without edits):

* ``register_model(name, model, metrics, feature_list) -> version:int``
* ``list_versions(name) -> list[dict]``
* ``load_model(name, version) -> (model, metadata)``
* ``load_best_model(name) -> (model, metadata)``   - best = lowest test RMSE

Each version is a Hopsworks *python* model whose uploaded artifact directory
holds:

* ``model.joblib``   - the fitted estimator (``joblib.dump``)
* ``metadata.json``  - the full record ``{name, version, created_at,
  model_class, metrics, feature_list}``, with ``metrics`` kept in the original
  nested shape (``{"val": {...}, "test": {...}, "algorithm": ...}``).

The Hopsworks model's own ``metrics`` field is a *flat numeric* subset
(``test_rmse``/``val_rmse``/...) used only for the registry UI and for picking
"best" without downloading every artifact. ``metadata.json`` inside the artifact
is the source of truth that :func:`load_model` / :func:`list_versions` read back,
so those return exactly what the local implementation did.

Always backed by the real Hopsworks project (``HOPSWORKS_API_KEY`` /
``HOPSWORKS_PROJECT_NAME``) - there is no local fallback.
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path
from typing import Any

import joblib

from aqi_predictor import hopsworks_client

_MODEL_FILE = "model.joblib"
_META_FILE = "metadata.json"

_FLAT_METRIC_KEYS = ("rmse", "mae", "r2")


def _project():
    """Hopsworks project handle. Separate function so tests can monkeypatch it."""
    return hopsworks_client.get_project()


def _model_registry():
    return _project().get_model_registry()


def _flat_metrics(metrics: dict) -> dict[str, float]:
    """Flat numeric ``{split}_{metric}`` dict for the Hopsworks registry UI.

    The nested ``metrics`` passed to :func:`register_model` also carries
    non-numeric keys (``algorithm``, ``selected_by``, ...) which Hopsworks'
    ``create_model(metrics=...)`` does not want; this pulls out just the numbers.
    """
    flat: dict[str, float] = {}
    for split in ("val", "test"):
        sub = metrics.get(split) if isinstance(metrics, dict) else None
        if not isinstance(sub, dict):
            continue
        for key in _FLAT_METRIC_KEYS:
            if key in sub:
                try:
                    flat[f"{split}_{key}"] = float(sub[key])
                except (TypeError, ValueError):
                    pass
    return flat


def _test_rmse(metadata: dict) -> float:
    """Sort key for 'best'; missing metric sorts last."""
    try:
        return float(metadata["metrics"]["test"]["rmse"])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def _load_artifact_dir(path: Path) -> tuple[Any, dict]:
    model = joblib.load(path / _MODEL_FILE)
    metadata = json.loads((path / _META_FILE).read_text(encoding="utf-8"))
    return model, metadata


def _download(hops_model) -> tuple[Any, dict]:
    """Download one Hopsworks model version and return ``(estimator, metadata)``.

    ``metadata.json`` was uploaded before Hopsworks assigned a version, so the
    real name/version from the registry object are overlaid on read.
    """
    directory = Path(hops_model.download())
    model, metadata = _load_artifact_dir(directory)
    metadata["name"] = hops_model.name
    metadata["version"] = int(hops_model.version)
    return model, metadata


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def register_model(
    name: str,
    model: Any,
    metrics: dict,
    feature_list: list[str],
) -> int:
    """Persist ``model`` as the next version of ``name``; return the version int."""
    mr = _model_registry()

    metadata = {
        "name": name,
        # Placeholder: Hopsworks assigns the real version on save(); load_model /
        # list_versions overlay the assigned version when they read this back.
        "version": None,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "metrics": metrics,
        "feature_list": list(feature_list),
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        joblib.dump(model, tmp_path / _MODEL_FILE)
        (tmp_path / _META_FILE).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        hops_model = mr.python.create_model(
            name=name,
            metrics=_flat_metrics(metrics),
            description=f"AQI regressor '{name}' ({metadata['model_class']})",
        )
        hops_model.save(str(tmp_path))

    return int(hops_model.version)


def list_versions(name: str) -> list[dict]:
    """Metadata for every registered version of ``name`` (oldest first)."""
    models = _model_registry().get_models(name)
    out = []
    for hops_model in sorted(models, key=lambda m: int(m.version)):
        _model, metadata = _download(hops_model)
        out.append(metadata)
    return out


def load_model(name: str, version: int) -> tuple[Any, dict]:
    hops_model = _model_registry().get_model(name, version=version)
    if hops_model is None:
        raise FileNotFoundError(f"no model {name!r} version {version}")
    return _download(hops_model)


def load_best_model(name: str) -> tuple[Any, dict]:
    """Return ``(model, metadata)`` for the version with the lowest test RMSE."""
    models = _model_registry().get_models(name)
    if not models:
        raise FileNotFoundError(f"no registered models named {name!r}")

    def _flat_test_rmse(hops_model) -> float:
        metric_dict = getattr(hops_model, "training_metrics", None) or {}
        try:
            return float(metric_dict["test_rmse"])
        except (KeyError, TypeError, ValueError):
            return float("inf")

    scored = [(m, _flat_test_rmse(m)) for m in models]
    if any(score != float("inf") for _m, score in scored):
        best_model = min(scored, key=lambda pair: pair[1])[0]
        return _download(best_model)

    # SDK did not expose flat metrics without a download - compare artifacts.
    best_meta = min(list_versions(name), key=_test_rmse)
    return load_model(name, best_meta["version"])
