"""Local model registry with a Hopsworks-shaped interface.

Same "local fallback" pattern as :mod:`aqi_predictor.feature_pipeline.store`: a
later phase can swap the internals for the Hopsworks Model Registry without
touching call sites.

* ``register_model(name, model, metrics, feature_list) -> version``
* ``load_best_model(name) -> (model, metadata)``   - best = lowest test RMSE

Layout (``models/`` is git-ignored)::

    models/<name>/v<version>/model.joblib
    models/<name>/v<version>/metadata.json
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib

from aqi_predictor.config import MODELS_DIR

_MODEL_FILE = "model.joblib"
_META_FILE = "metadata.json"


def _model_dir(name: str) -> Path:
    return MODELS_DIR / name


def _versions(name: str) -> list[int]:
    root = _model_dir(name)
    if not root.exists():
        return []
    out = []
    for p in root.glob("v*"):
        if p.is_dir() and (p / _META_FILE).exists():
            try:
                out.append(int(p.name[1:]))
            except ValueError:
                continue
    return sorted(out)


def _test_rmse(metadata: dict) -> float:
    """Sort key for 'best'; missing metric sorts last."""
    try:
        return float(metadata["metrics"]["test"]["rmse"])
    except (KeyError, TypeError, ValueError):
        return float("inf")


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
    version = (_versions(name)[-1] + 1) if _versions(name) else 1
    vdir = _model_dir(name) / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, vdir / _MODEL_FILE)
    metadata = {
        "name": name,
        "version": version,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "metrics": metrics,
        "feature_list": list(feature_list),
    }
    (vdir / _META_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return version


def list_versions(name: str) -> list[dict]:
    """Metadata for every registered version of ``name`` (oldest first)."""
    out = []
    for v in _versions(name):
        meta = _model_dir(name) / f"v{v}" / _META_FILE
        out.append(json.loads(meta.read_text(encoding="utf-8")))
    return out


def load_model(name: str, version: int) -> tuple[Any, dict]:
    vdir = _model_dir(name) / f"v{version}"
    if not (vdir / _MODEL_FILE).exists():
        raise FileNotFoundError(f"no model {name!r} version {version}")
    model = joblib.load(vdir / _MODEL_FILE)
    metadata = json.loads((vdir / _META_FILE).read_text(encoding="utf-8"))
    return model, metadata


def load_best_model(name: str) -> tuple[Any, dict]:
    """Return ``(model, metadata)`` for the version with the lowest test RMSE."""
    versions = list_versions(name)
    if not versions:
        raise FileNotFoundError(f"no registered models named {name!r}")
    best = min(versions, key=_test_rmse)
    return load_model(name, best["version"])
