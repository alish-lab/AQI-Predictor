"""Historical backfill for the AQI feature pipeline.

For every location in ``config.LOCATIONS``:

1. fetch hourly air-quality + weather from Open-Meteo, month by month,
2. engineer features + handle missing data,
3. upsert the result into the local feature store,

then print / save a data-quality report.

Usage::

    python -m scripts.backfill                     # all locations, full history
    python scripts/backfill.py --location karachi --start 2025-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from aqi_predictor.config import (
    BACKFILL_START_DATE,
    FEATURE_STORE_DIR,
    LOCATIONS,
    POLLUTANT_VARS,
    RAW_DIR,
    WEATHER_VARS,
    get_location,
)
from aqi_predictor.feature_pipeline import fetch, store
from aqi_predictor.feature_pipeline.features import build_features


def _month_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    chunks: list[tuple[dt.date, dt.date]] = []
    s = start
    while s <= end:
        first_next = (s.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        chunks.append((s, min(first_next - dt.timedelta(days=1), end)))
        s = first_next
    return chunks


def _fetch_history(location: dict, start: dt.date, end: dt.date) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for c_start, c_end in _month_chunks(start, end):
        print(f"    {c_start} -> {c_end}", flush=True)
        parts.append(fetch.historical(location, c_start, c_end))
    raw = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["location", "time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return raw


def backfill_location(location: dict, start: dt.date, end: dt.date) -> dict:
    print(f"[{location['name']}] fetching {start} -> {end}", flush=True)
    raw = _fetch_history(location, start, end)

    raw_path = RAW_DIR / f"{location['name']}.parquet"
    raw.to_parquet(raw_path, index=False)
    print(f"[{location['name']}] raw merged pull: {len(raw)} rows -> {raw_path}", flush=True)

    features, reports = build_features(raw)
    store.insert_features(features)
    print(
        f"[{location['name']}] engineered {features.shape[1]} cols, "
        f"upserted {len(features)} rows",
        flush=True,
    )
    # one location in, one report out
    return reports[0].to_dict() if reports else {"location": location["name"]}


def _store_summary(locations: list[str]) -> dict:
    """Cross-check what actually landed in the feature store."""
    view = store.get_feature_view("2000-01-01", "2100-01-01", locations=locations)
    if view.empty:
        return {"n_rows": 0}

    base_cols = ["location", "time", *POLLUTANT_VARS, *WEATHER_VARS]
    present = [c for c in base_cols if c in view.columns and c not in ("location", "time")]
    per_loc = {
        loc: {
            "n_rows": int(len(g)),
            "time_min": str(g["time"].min()),
            "time_max": str(g["time"].max()),
        }
        for loc, g in view.groupby("location")
    }
    return {
        "n_rows": int(len(view)),
        "n_columns": int(view.shape[1]),
        "time_min": str(view["time"].min()),
        "time_max": str(view["time"].max()),
        "missing_pct_final": {
            c: round(100.0 * view[c].isna().mean(), 3) for c in present
        },
        "per_location": per_loc,
    }


def _print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("BACKFILL DATA-QUALITY REPORT")
    print("=" * 70)
    print(f"generated_at : {report['generated_at']}")
    print(f"range        : {report['start']} -> {report['end']}")

    store_s = report["feature_store"]
    print("\n-- feature store --")
    print(f"rows         : {store_s.get('n_rows')}")
    print(f"columns      : {store_s.get('n_columns')}")
    print(f"coverage     : {store_s.get('time_min')} -> {store_s.get('time_max')}")
    for loc, s in store_s.get("per_location", {}).items():
        print(f"  {loc:<12} {s['n_rows']:>6} rows  {s['time_min']} -> {s['time_max']}")

    for loc_report in report["locations"]:
        print(f"\n-- {loc_report['location']} : missing-data handling --")
        print(
            f"input rows {loc_report.get('n_input_rows')}, "
            f"hourly grid {loc_report.get('n_hourly_grid_rows')}, "
            f"interpolated cells {loc_report.get('n_cells_interpolated_total')}, "
            f"rows dropped {loc_report.get('n_rows_dropped')}, "
            f"output rows {loc_report.get('n_output_rows')}"
        )
        before = loc_report.get("missing_pct_before", {})
        after = loc_report.get("missing_pct_after", {})
        interp = loc_report.get("cells_interpolated", {})
        print(f"  {'column':<22}{'% miss before':>14}{'% miss after':>14}{'interp cells':>14}")
        for col in before:
            print(
                f"  {col:<22}{before.get(col, 0):>14}{after.get(col, 0):>14}"
                f"{interp.get(col, 0):>14}"
            )
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", action="append", help="location name(s); default: all")
    parser.add_argument("--start", default=BACKFILL_START_DATE, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    # Open-Meteo works in UTC; the local date can be a day ahead of its "today".
    utc_today = dt.datetime.now(dt.timezone.utc).date()
    end = dt.date.fromisoformat(args.end) if args.end else utc_today
    locations = (
        [get_location(n) for n in args.location] if args.location else list(LOCATIONS)
    )

    loc_reports = [backfill_location(loc, start, end) for loc in locations]

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "locations": loc_reports,
        "feature_store": _store_summary([loc["name"] for loc in locations]),
    }

    report_path = FEATURE_STORE_DIR / "backfill_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"report saved -> {report_path}")


if __name__ == "__main__":
    main()
