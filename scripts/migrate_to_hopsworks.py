"""One-off: push the existing local feature-store parquet(s) into Hopsworks.

Phase 1's backfill wrote correctly-computed hourly rows to
``data/feature_store/location=*/data.parquet`` from real Open-Meteo pulls (Karachi:
18,336 rows, 2024-07-29 -> now). Phase 4.5 switched
:mod:`aqi_predictor.feature_pipeline.store` to a real Hopsworks feature group.
Re-running the full historical backfill through ``fetch.py`` would just re-derive
the same numbers slower (and risk rate limits), so this reads the parquet files
that already exist and inserts them once via the new ``store.insert_features``.

Not wired into any pipeline. Run manually, once::

    python scripts/migrate_to_hopsworks.py
"""

from __future__ import annotations

import sys

import pandas as pd

from aqi_predictor.config import FEATURE_STORE_DIR
from aqi_predictor.feature_pipeline import store


def main() -> int:
    parquets = sorted(FEATURE_STORE_DIR.glob("location=*/data.parquet"))
    if not parquets:
        print(f"no parquet files found under {FEATURE_STORE_DIR}", file=sys.stderr)
        return 1

    frames = []
    for path in parquets:
        location = path.parent.name.split("=", 1)[1]
        part = pd.read_parquet(path)
        print(
            f"read {len(part):>7} rows  location={location:<12} "
            f"{part['time'].min()} .. {part['time'].max()}  ({path})"
        )
        frames.append(part)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\ninserting {len(combined)} combined rows into the "
          f"'{store.FEATURE_GROUP_NAME}' feature group ...")
    store.insert_features(combined)

    print("\nreading it back for a sanity check ...")
    view = store.get_feature_view("2000-01-01", "2100-01-01")
    print(f"feature view: {len(view)} rows, {view.shape[1]} columns")
    for location, group in view.groupby("location"):
        print(
            f"  {location:<12} {len(group):>7} rows  "
            f"{group['time'].min()} .. {group['time'].max()}"
        )

    local_total = sum(len(pd.read_parquet(p)) for p in parquets)
    if len(view) == local_total:
        print(f"\nOK - round-trip row count matches the local parquet(s) ({local_total}).")
        return 0
    print(
        f"\nWARNING - feature view has {len(view)} rows but the local parquet(s) "
        f"hold {local_total}. Investigate before relying on the migration.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
