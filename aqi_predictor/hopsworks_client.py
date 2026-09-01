"""Cached Hopsworks project handle, shared by the feature store and model registry.

``hopsworks.login`` opens an authenticated session against the configured
serverless project. It is comparatively slow and there is no reason to do it more
than once per process, so the handle is created lazily and memoised here.

Both :mod:`aqi_predictor.feature_pipeline.store` and
:mod:`aqi_predictor.training_pipeline.registry` call this through their own
module-level ``_project`` indirection, which the smoke tests monkeypatch with an
in-memory fake so they never touch the network.
"""

from __future__ import annotations

from pathlib import Path

import hopsworks

from aqi_predictor import config

_PROJECT = None


def _ensure_tmp_dir() -> None:
    """Create a ``/tmp`` directory the hopsworks client can write to.

    Parts of ``hopsworks_common`` (notably the Kafka storage connector's PEM
    export in ``client/base.py::_write_pem``) hardcode ``os.path.join("/tmp",
    ...)`` with no override hook. On Windows that resolves to ``<cwd-drive>:\\tmp``
    which does not exist by default, so ``fg.insert`` blows up with
    ``FileNotFoundError``. Creating it once here is the least-bad fix until the
    upstream paths respect a configurable temp dir.
    """
    Path("/tmp").mkdir(parents=True, exist_ok=True)


def get_project():
    """Return the shared, memoised Hopsworks project handle."""
    global _PROJECT
    if _PROJECT is None:
        if not config.HOPSWORKS_API_KEY or not config.HOPSWORKS_PROJECT_NAME:
            raise RuntimeError(
                "HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME are not set; add them "
                "to the .env file at the repo root."
            )
        # cert_folder defaults to "/tmp", which os.mkdir cannot create on
        # Windows; point it at a real, writable path inside the repo instead
        # (gitignored). With a non-default value the client uses it directly.
        _ensure_tmp_dir()
        cert_folder = config.PROJECT_ROOT / ".hopsworks_certs"
        cert_folder.mkdir(parents=True, exist_ok=True)
        _PROJECT = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
            cert_folder=str(cert_folder),
        )
    return _PROJECT
