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
from aqi_predictor.feature_pipeline.features import (
    MAX_ROLLING_WINDOW_HOURS,
    build_features,
)

# Days of history to pull (and upsert) each run. The longest backward-looking
# feature is a 24h rolling window; a few days of context leaves the recent rows
# fully defined even if Open-Meteo is briefly missing the last hour or two.
DEFAULT_HISTORY_DAYS = 4

# Extra days of raw history fetched *before* the DEFAULT_HISTORY_DAYS target
# window, purely to seed rolling-window context - never upserted themselves.
# Without this, the earliest MAX_ROLLING_WINDOW_HOURS (24h) of *every* run's
# own fetch has no prior data to roll over within that fetch, so
# build_features correctly returns NaN for e.g. us_aqi_roll_mean_24h on those
# rows - not because the data is actually missing, but because this run only
# looked back DEFAULT_HISTORY_DAYS. Upserting that NaN then clobbers a
# perfectly good value an earlier run (which had more trailing context at the
# time) had already stored for the same (location, time) key, since
# Hopsworks/Hudi upsert replaces the whole row. One day comfortably covers the
# 24h rolling window with room to spare.
ROLLING_CONTEXT_BUFFER_DAYS = 1
assert ROLLING_CONTEXT_BUFFER_DAYS * 24 >= MAX_ROLLING_WINDOW_HOURS


def run_location(location: dict, days: int) -> dict:
    """Fetch -> build_features -> upsert for one location. Returns a summary dict.

    Fetches ``days + ROLLING_CONTEXT_BUFFER_DAYS`` of raw history so every row
    in the intended ``days``-wide upsert window gets a full rolling-window
    lookback, but only *upserts* the ``days``-wide window itself - the extra
    buffer rows exist solely to give those rows real context, not to be stored
    (their own features would be just as context-starved, one day earlier).
    """
    now = dt.datetime.now(dt.timezone.utc)
    upsert_start = now - dt.timedelta(days=days)
    fetch_start = (upsert_start - dt.timedelta(days=ROLLING_CONTEXT_BUFFER_DAYS)).date()
    end = now.date()
    name = location["name"]

    print(
        f"[{name}] fetching {fetch_start} -> {end} ({days}d target window + "
        f"{ROLLING_CONTEXT_BUFFER_DAYS}d rolling-context buffer) ...",
        flush=True,
    )
    raw = fetch.historical(location, fetch_start, end)
    print(f"[{name}] fetched {len(raw)} hourly rows", flush=True)

    featured, _reports = build_features(raw)
    if featured.empty:
        print(f"[{name}] build_features produced no rows; nothing to upsert", flush=True)
        return {"location": name, "fetched": len(raw), "upserted": 0}

    # Drop the buffer rows now that they've done their job (giving the real
    # target window full rolling-window context) - only the target window is
    # upserted, so a buffer row's own (possibly context-starved) features never
    # overwrite a better-contexted value already stored for it.
    features = featured[featured["time"] >= upsert_start].reset_index(drop=True)
    if features.empty:
        print(
            f"[{name}] no rows left in the target window after buffering; "
            f"nothing to upsert",
            flush=True,
        )
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
