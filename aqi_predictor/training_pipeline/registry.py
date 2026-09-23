"""Hopsworks Model Registry integration for the training pipeline.

Public API (unchanged from the earlier local-joblib implementation - every call
site keeps working without edits):

* ``register_model(name, model, metrics, feature_list) -> version:int``
* ``list_versions(name) -> list[dict]``
* ``load_model(name, version) -> (model, metadata)``
* ``load_best_model(name) -> (model, metadata)``   - best = lowest test RMSE
* ``current_model_metrics(names) -> list[dict]``   - a one-time snapshot of
  each name's served version's test metrics (RMSE/MAE/R2, algorithm, version),
  via ``load_best_model``; used by ``scripts/evaluate_baselines.py`` and the
  dashboard's "Model Performance" table so neither re-implements the loop

Each version is a Hopsworks *python* model whose uploaded artifact directory
holds:

* ``model.joblib`` **or** ``model.xgb.ubj`` - the fitted estimator. XGBoost
  models (``XGBRegressor``) are saved with their own ``save_model()`` (UBJSON)
  instead of ``joblib.dump``: pickling an ``XGBRegressor`` embeds its raw
  booster buffer, and XGBoost's docs are explicit that this is *not*
  guaranteed cross-version/cross-platform-safe - unlike ``save_model()``/
  ``load_model()``, which is. This was found the hard way: every
  ``us_aqi_h72`` XGBoost version registered from v3 onward (and 2 recent
  ``us_aqi_next`` ones) became undeserializable pickle noise
  (``XGBoostError: input stream corrupted``) despite the exact same pinned
  ``xgboost==3.3.0`` everywhere. ``metadata.json``'s ``model_file`` field
  records which filename/format a given version used, so old joblib-based
  artifacts (including the already-corrupted ones - not repaired, not
  repairable from a bad pickle) keep loading exactly as before. Ridge and
  RandomForest (plain sklearn, no custom binary buffer) and the LSTM
  (``torch`` state via ``joblib``) do not have this problem - confirmed via
  the live registry, every RF/LSTM version loaded fine - so their persistence
  is unchanged.
* ``metadata.json``  - the full record ``{name, version, created_at,
  model_class, model_file, metrics, feature_list}``, with ``metrics`` kept in
  the original nested shape (``{"val": {...}, "test": {...}, "algorithm":
  ...}``).
* any files named in ``metadata["extra_artifact_files"]`` - additional
  ``joblib``-dumped objects (e.g. a fitted ``StandardScaler`` an LSTM needs at
  inference time) passed via ``register_model(..., extra_artifacts=...)``.
  ``load_model`` / ``load_best_model`` load these back and attach them to the
  returned metadata dict as ``metadata["extra_artifacts"] = {filename: obj}``.
  Models registered without ``extra_artifacts`` (the four tree/linear models)
  get back an empty dict, unchanged from before this existed.

``load_best_model`` picks the lowest-test-RMSE version *that actually
deserializes and clears a test-R2 floor* (:data:`_MIN_ACCEPTABLE_TEST_R2`): it
tries candidates in ascending-RMSE order and falls back to the next-best on
either a load failure or an at-or-below-floor R2 (both loudly ``print``-logged,
matching this codebase's existing style rather than the stdlib ``logging``
module, which nothing else here uses), instead of crashing the caller or
silently serving a degenerate model. Previously a dashboard/inference request
for a horizon whose best-by-metric version was a corrupted artifact would
hard-crash (``us_aqi_h72``); separately, a version with the numerically-lowest
RMSE but a negative R2 (evaluated against too small/atypical a test split for
RMSE alone to mean anything - seen live: ``us_aqi_h24`` v8) would be served
without complaint. The R2 floor only ever disqualifies a candidate - it never
reorders the RMSE ranking among versions that pass it. ``list_versions`` skips
and logs any version that fails to load (not the R2 floor - that check only
applies to *selecting* a "best" version) rather than aborting the whole
listing, since ``scripts/check_registry_after_training.py`` and
``load_best_model``'s own no-flat-metrics fallback both depend on it. An
explicit ``load_model(name, version)`` for one exact version is unchanged -
it still raises if that specific version is corrupted, and it never applies
the R2 floor, since silently substituting a different version when the caller
asked for one by number would be the wrong kind of "safe".

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
from xgboost import XGBRegressor

from aqi_predictor import hopsworks_client

_MODEL_FILE = "model.joblib"
# XGBoost's own persistence format (UBJSON) - portable across xgboost versions
# and platforms, unlike a joblib/pickle of the raw booster buffer. See the
# module docstring for why this exists.
_XGBOOST_MODEL_FILE = "model.xgb.ubj"
_META_FILE = "metadata.json"

_FLAT_METRIC_KEYS = ("rmse", "mae", "r2")

# Decimal places test RMSE is rounded to before comparing "best" versions.
# RandomForest/XGBoost with n_jobs=-1 can produce run-to-run floating-point
# noise on the order of 1e-14 from parallel reduction order even with a fixed
# random_state and identical data - without rounding, that noise (not a real
# accuracy difference) breaks the version tie-break below.
_RMSE_COMPARISON_PRECISION = 6

# A version whose test R2 is at or below this is no better than predicting the
# mean (0.0) - never eligible to be auto-served no matter how low its RMSE,
# since a "good" RMSE alongside a negative R2 means the test split it was
# evaluated on was too small/atypical for RMSE alone to mean anything (seen
# live: us_aqi_h24 v8, RMSE 3.73 - nominally the best of 21 versions - but
# R2 -0.76). This is a floor, not a ranking signal: it only ever disqualifies
# a candidate, never reorders the RMSE-ranked list otherwise.
_MIN_ACCEPTABLE_TEST_R2 = 0.0


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


def _test_r2(metadata: dict) -> float | None:
    """Test R2 from full ``metadata.json``, or ``None`` if unavailable/unparseable.

    ``None`` (not a sentinel number) so the :data:`_MIN_ACCEPTABLE_TEST_R2`
    floor check can tell "known bad" apart from "unknown" - a version with no
    parseable R2 is passed through rather than penalized for missing data.
    """
    try:
        return float(metadata["metrics"]["test"]["r2"])
    except (KeyError, TypeError, ValueError):
        return None


def _load_artifact_dir(path: Path) -> tuple[Any, dict]:
    """Read ``metadata.json`` first (cheap, always readable) so it can tell us
    *which format the model file is in* before we try to load it - old
    artifacts (no ``model_file`` key) default to the original joblib path, so
    every RF/Ridge/LSTM version keeps loading exactly as before.
    """
    metadata = json.loads((path / _META_FILE).read_text(encoding="utf-8"))
    model_file = metadata.get("model_file", _MODEL_FILE)
    if model_file == _XGBOOST_MODEL_FILE:
        model = XGBRegressor()
        model.load_model(str(path / model_file))
    else:
        model = joblib.load(path / model_file)
    return model, metadata


def _download(hops_model) -> tuple[Any, dict]:
    """Download one Hopsworks model version and return ``(estimator, metadata)``.

    ``metadata.json`` was uploaded before Hopsworks assigned a version, so the
    real name/version from the registry object are overlaid on read.
    ``metadata["extra_artifacts"]`` is populated by loading every file named in
    ``metadata["extra_artifact_files"]`` (empty/missing for models registered
    without ``extra_artifacts``, i.e. all four tree/linear models).

    Raises whatever the underlying deserialization raises (e.g.
    ``xgboost.core.XGBoostError`` for a pre-fix corrupted artifact) - callers
    that need to skip a bad version instead of crashing use
    :func:`_safe_download`.
    """
    directory = Path(hops_model.download())
    model, metadata = _load_artifact_dir(directory)
    metadata["name"] = hops_model.name
    metadata["version"] = int(hops_model.version)
    extra_files = metadata.get("extra_artifact_files") or []
    metadata["extra_artifacts"] = {
        fname: joblib.load(directory / fname) for fname in extra_files
    }
    return model, metadata


def _safe_download(hops_model) -> tuple[Any, dict] | None:
    """``_download``, but a deserialization failure is logged loudly and
    returns ``None`` instead of raising - used anywhere a corrupted historical
    version (e.g. one of the pre-fix joblib-pickled XGBoost artifacts) should
    be skipped rather than take down the whole call.
    """
    try:
        return _download(hops_model)
    except Exception as exc:  # noqa: BLE001 - any deserialization failure, not just XGBoost's
        print(
            f"[registry] WARNING: {hops_model.name!r} v{hops_model.version} "
            f"failed to deserialize ({type(exc).__name__}: {exc}) - skipping "
            f"this version",
            flush=True,
        )
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def register_model(
    name: str,
    model: Any,
    metrics: dict,
    feature_list: list[str],
    shap_importance: dict[str, float] | None = None,
    extra_artifacts: dict[str, Any] | None = None,
) -> int:
    """Persist ``model`` as the next version of ``name``; return the version int.

    ``shap_importance`` is an optional ``{feature: mean_abs_shap_value}`` dict
    (tree-based models and the LSTM) stored in ``metadata.json`` for the
    dashboard's global feature-importance chart. ``None`` for models without a
    SHAP explainer (e.g. Ridge).

    ``extra_artifacts`` is an optional ``{filename: object}`` dict of
    additional objects to ``joblib.dump`` alongside ``model.joblib`` (e.g. the
    LSTM's ``{"scaler.joblib": scaler, "shap_background.joblib": background}``).
    Omitted (``None``) for the four tree/linear models - no behavior change for
    them.
    """
    mr = _model_registry()
    is_xgboost = isinstance(model, XGBRegressor)
    model_file = _XGBOOST_MODEL_FILE if is_xgboost else _MODEL_FILE

    metadata = {
        "name": name,
        # Placeholder: Hopsworks assigns the real version on save(); load_model /
        # list_versions overlay the assigned version when they read this back.
        "version": None,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        # Which file/format the model itself is stored in - read by
        # _load_artifact_dir before it tries to load anything. Old artifacts
        # (registered before this field existed) default to joblib on read.
        "model_file": model_file,
        "metrics": metrics,
        "feature_list": list(feature_list),
        "shap_importance": shap_importance,
        "extra_artifact_files": sorted(extra_artifacts) if extra_artifacts else [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if is_xgboost:
            # XGBoost's native format - portable across versions/platforms,
            # unlike joblib-pickling the raw booster buffer (see module docstring).
            model.save_model(str(tmp_path / model_file))
        else:
            joblib.dump(model, tmp_path / model_file)
        for filename, obj in (extra_artifacts or {}).items():
            joblib.dump(obj, tmp_path / filename)
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
    """Metadata for every *loadable* registered version of ``name`` (oldest
    first). A version whose artifact fails to deserialize (e.g. a pre-fix
    joblib-pickled XGBoost model - see the module docstring) is skipped, not
    raised - logged loudly via :func:`_safe_download` instead. Used by
    ``scripts/check_registry_after_training.py`` and by
    :func:`load_best_model`'s no-flat-metrics fallback, both of which need a
    listing that a single corrupted version can't take down.
    """
    models = sorted(_model_registry().get_models(name), key=lambda m: int(m.version))
    out = []
    skipped = 0
    for hops_model in models:
        result = _safe_download(hops_model)
        if result is None:
            skipped += 1
            continue
        _model, metadata = result
        out.append(metadata)
    if skipped:
        print(
            f"[registry] {name!r}: {skipped}/{len(models)} version(s) skipped "
            f"(failed to deserialize), {len(out)} returned",
            flush=True,
        )
    return out


def load_model(name: str, version: int) -> tuple[Any, dict]:
    hops_model = _model_registry().get_model(name, version=version)
    if hops_model is None:
        raise FileNotFoundError(f"no model {name!r} version {version}")
    return _download(hops_model)


def load_best_model(name: str) -> tuple[Any, dict]:
    """Return ``(model, metadata)`` for the best-test-RMSE version *that
    actually deserializes*.

    Test RMSE is rounded to :data:`_RMSE_COMPARISON_PRECISION` decimals before
    comparing, so run-to-run floating-point noise (e.g. from RandomForest's
    parallel reduction order) doesn't masquerade as a real accuracy
    difference. Ties after rounding break toward the *higher* version number,
    so a newer version - such as one with ``shap_importance`` added - reliably
    wins over an older tied one instead of depending on API return order.

    Candidates are tried in ascending-RMSE order; a candidate is skipped, in
    favor of the next-best one, if *either*:

    * it fails to deserialize (e.g. a pre-fix joblib-pickled XGBoost artifact -
      see the module docstring) - this is what previously took the whole
      request down for ``us_aqi_h72``, whose lowest-recorded-RMSE version was
      an unloadable XGBoost model, or
    * its test R2 is at or below :data:`_MIN_ACCEPTABLE_TEST_R2` - a floor, not
      a ranking change: RMSE still decides the order among versions that pass
      it. This is what let ``us_aqi_h24`` v8 (RMSE 3.73, nominally the best of
      21 versions, but R2 -0.76 - no better than predicting the mean) get
      auto-served over v6/v10, both with R2 > 0.85 and RMSE only marginally
      higher. A version with no parseable R2 at all is passed through rather
      than penalized for missing data - this is a floor against a *known-bad*
      value, not a requirement that one exist.

    Both cases are logged loudly. Only raises if every registered version is
    rejected for one reason or the other.
    """
    models = _model_registry().get_models(name)
    if not models:
        raise FileNotFoundError(f"no registered models named {name!r}")

    def _flat_test_rmse(hops_model) -> float:
        metric_dict = getattr(hops_model, "training_metrics", None) or {}
        try:
            return float(metric_dict["test_rmse"])
        except (KeyError, TypeError, ValueError):
            return float("inf")

    def _flat_test_r2(hops_model) -> float | None:
        metric_dict = getattr(hops_model, "training_metrics", None) or {}
        try:
            return float(metric_dict["test_r2"])
        except (KeyError, TypeError, ValueError):
            return None

    scored = [(m, _flat_test_rmse(m)) for m in models]
    if any(score != float("inf") for _m, score in scored):
        ranked = sorted(
            scored,
            key=lambda pair: (round(pair[1], _RMSE_COMPARISON_PRECISION), -int(pair[0].version)),
        )
        candidates = [m for m, _score in ranked]
    else:
        # SDK did not expose flat metrics without a download - compare
        # artifacts. list_versions() already skips unloadable versions, so
        # whatever it returns is safe to load again here.
        loadable = [
            meta
            for meta in list_versions(name)
            if not (
                (r2 := _test_r2(meta)) is not None and r2 <= _MIN_ACCEPTABLE_TEST_R2
            )
        ]
        if not loadable:
            raise FileNotFoundError(
                f"no loadable version of {name!r} passes the test-R2 floor "
                f"(> {_MIN_ACCEPTABLE_TEST_R2})"
            )
        best_meta = min(
            loadable,
            key=lambda m: (round(_test_rmse(m), _RMSE_COMPARISON_PRECISION), -int(m["version"])),
        )
        return load_model(name, best_meta["version"])

    last_error: Exception | None = None
    for hops_model in candidates:
        r2 = _flat_test_r2(hops_model)
        if r2 is not None and r2 <= _MIN_ACCEPTABLE_TEST_R2:
            print(
                f"[registry] WARNING: {name!r} v{hops_model.version} (best-by-RMSE "
                f"candidate) has test R2={r2:.4f} (<= {_MIN_ACCEPTABLE_TEST_R2}, no "
                f"better than predicting the mean) - rejecting in favor of the "
                f"next-best version",
                flush=True,
            )
            continue
        try:
            return _download(hops_model)
        except Exception as exc:  # noqa: BLE001 - any deserialization failure
            print(
                f"[registry] WARNING: {name!r} v{hops_model.version} (best-by-RMSE "
                f"candidate) failed to deserialize ({type(exc).__name__}: {exc}) - "
                f"falling back to the next-best version",
                flush=True,
            )
            last_error = exc

    raise RuntimeError(
        f"all {len(candidates)} registered version(s) of {name!r} were rejected "
        f"(failed to deserialize, or test R2 <= {_MIN_ACCEPTABLE_TEST_R2}); "
        f"last deserialization error: {last_error!r}"
    ) from last_error


def current_model_metrics(names: list[str]) -> list[dict]:
    """Test RMSE/MAE/R2 + algorithm/version for the currently-served (best)
    version of each name in ``names``, via :func:`load_best_model`.

    A one-time snapshot of whatever's live right now, not a historical trend -
    no querying beyond one ``load_best_model`` call per name. Any name with no
    registered version at all, or whose every registered version fails to
    deserialize (``load_best_model`` exhausts its fallback and raises
    ``RuntimeError`` - see there), is silently skipped here (callers that need
    to report that decide how to phrase it); the returned list is otherwise in
    the same order as ``names``.
    """
    rows: list[dict] = []
    for name in names:
        try:
            _model, meta = load_best_model(name)
        except (FileNotFoundError, RuntimeError):
            continue
        metrics = meta.get("metrics") or {}
        test_metrics = metrics.get("test") or {}
        rows.append(
            {
                "name": name,
                "version": meta["version"],
                "algorithm": metrics.get("algorithm", "ml"),
                "rmse": test_metrics.get("rmse"),
                "mae": test_metrics.get("mae"),
                "r2": test_metrics.get("r2"),
            }
        )
    return rows
