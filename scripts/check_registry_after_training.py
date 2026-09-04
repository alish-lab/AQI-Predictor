"""Log which registered model version ``predict.py`` now serves for each name.

``registry.load_best_model`` always returns the lowest-test-RMSE version across
*every* version ever registered, so a weak daily training run cannot regress what
the inference pipeline serves - it just won't be selected. This script makes that
outcome visible in the training job log: for each model name it prints how many
versions exist, which one is the newest (i.e. the one today's run just
registered), and which one ``load_best_model`` picks.

It gates nothing - there is deliberately no promotion logic here or in
``registry.register_model``.

Run::

    python -m scripts.check_registry_after_training
    python -m scripts.check_registry_after_training --name us_aqi_next
"""

from __future__ import annotations

import argparse

from aqi_predictor.training_pipeline import registry

DEFAULT_NAMES = ["us_aqi_next", "us_aqi_h24", "us_aqi_h48", "us_aqi_h72"]


def _test_rmse(metadata: dict) -> float | None:
    try:
        return float(metadata["metrics"]["test"]["rmse"])
    except (KeyError, TypeError, ValueError):
        return None


def report(name: str) -> None:
    versions = registry.list_versions(name)
    if not versions:
        print(f"[{name}] no registered versions", flush=True)
        return

    latest = max(int(v["version"]) for v in versions)
    _model, best = registry.load_best_model(name)
    best_version = int(best["version"])
    rmse = _test_rmse(best)
    rmse_str = f"{rmse:.4f}" if rmse is not None else "n/a"

    verdict = (
        "today's new version WINS"
        if best_version == latest
        else f"an older version (v{best_version}) still wins"
    )
    print(
        f"[{name}] {len(versions)} version(s); newest=v{latest}; "
        f"serving=v{best_version} (test RMSE {rmse_str}) -> {verdict}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", action="append", help="model name(s); default: all four"
    )
    args = parser.parse_args()

    for name in args.name or DEFAULT_NAMES:
        report(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
