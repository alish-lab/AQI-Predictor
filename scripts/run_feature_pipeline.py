"""Hourly feature pipeline: fetch recent data, engineer features, upsert to Hopsworks.

For every location in :data:`aqi_predictor.config.LOCATIONS`:

1. fetch the last few days of hourly air-quality + weather from Open-Meteo via
   ``fetch.historical`` - **not** ``fetch.latest_hour``: the lag, rolling and
   change-rate features need days of prior context, a single row can't produce
   them;
2. run :func:`build_features` (the same engineering the backfill and the feature
   store use);
3. ``store.insert_features`` - upsert into the ``aqi_features`` feature group.
   It is keyed on ``(location, time)``, so re-running over an overlapping window
   simply overwrites those hours (idempotent).

Run::

    python -m scripts.run_feature_pipeline                       # all locations
    python scripts/run_feature_pipeline.py --location karachi --days 7
"""

from __future__ import annotations

import argparse
import datetime as dt

from aqi_predictor.config import LOCATIONS, get_location
from aqi_predictor.feature_pipeline import fetch, store
from aqi_predictor.feature_pipeline.features import build_features

# Days of history to pull each run. The longest backward-looking feature is a
# 24h rolling window; a few days of context leaves the recent rows fully defined
# even if Open-Meteo is briefly missing the last hour or two.
DEFAULT_HISTORY_DAYS = 4


def run_location(location: dict, days: int) -> dict:
    """Fetch -> build_features -> upsert for one location. Returns a summary dict."""
    now = dt.datetime.now(dt.timezone.utc)
    start = (now - dt.timedelta(days=days)).date()
    end = now.date()
    name = location["name"]

    print(f"[{name}] fetching {start} -> {end} ...", flush=True)
    raw = fetch.historical(location, start, end)
    print(f"[{name}] fetched {len(raw)} hourly rows", flush=True)

    features, _reports = build_features(raw)
    if features.empty:
        print(f"[{name}] build_features produced no rows; nothing to upsert", flush=True)
        return {"location": name, "fetched": len(raw), "upserted": 0}

    store.insert_features(features)
    print(
        f"[{name}] engineered {features.shape[1]} cols, upserted {len(features)} rows "
        f"({features['time'].min()} -> {features['time'].max()})",
        flush=True,
    )
    return {"location": name, "fetched": len(raw), "upserted": len(features)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--location", action="append", help="location name(s); default: all"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"days of history to fetch (default: {DEFAULT_HISTORY_DAYS})",
    )
    args = parser.parse_args()

    locations = (
        [get_location(n) for n in args.location] if args.location else list(LOCATIONS)
    )

    summaries = [run_location(loc, args.days) for loc in locations]

    print("\n" + "=" * 60)
    print("FEATURE PIPELINE RUN SUMMARY")
    print("=" * 60)
    print(f"{'location':<16}{'fetched':>10}{'upserted':>10}")
    print("-" * 36)
    for s in summaries:
        print(f"{s['location']:<16}{s['fetched']:>10}{s['upserted']:>10}")
    print("=" * 60)

    if all(s["upserted"] == 0 for s in summaries):
        print("no rows upserted for any location", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
