"""Fetch hourly air-quality and weather data from Open-Meteo and merge them.

Open-Meteo splits what we need across two (sometimes three) endpoints:

* ``air-quality-api``  - pollutants + ``us_aqi``. Fresh to the current hour.
* ``archive-api``      - historical weather (ERA5). Lags ~1-2 days behind "now".
* ``api`` (forecast)   - weather with ``past_days``; used only to fill the gap
  between the archive cutoff and now.

Public API:

* ``historical(location, start_date, end_date)`` - full hourly history for a range.
* ``latest_hour(location)``                      - the most recent complete hour.

Both return a DataFrame with columns ``["location", "time", *POLLUTANT_VARS,
*WEATHER_VARS]`` where ``time`` is tz-aware UTC.
"""

from __future__ import annotations

import datetime as dt
import re
import time as _time
from typing import Any

import pandas as pd
import requests

from aqi_predictor.config import (
    AIR_QUALITY_URL,
    POLLUTANT_VARS,
    WEATHER_ARCHIVE_URL,
    WEATHER_FORECAST_URL,
    WEATHER_VARS,
    get_location,
)

# archive-api (ERA5) can lag "now" by a day or more. Whatever it doesn't cover
# yet is filled from the forecast endpoint's ``past_days`` window.


def _utc_today() -> dt.date:
    """Open-Meteo works in UTC; the local calendar date can be a day ahead."""
    return dt.datetime.now(dt.timezone.utc).date()


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "aqi-predictor/0.1 (+feature_pipeline)"})

COLUMN_ORDER = ["location", "time", *POLLUTANT_VARS, *WEATHER_VARS]


# --------------------------------------------------------------------------- #
# Low-level HTTP
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict[str, Any], *, retries: int = 4) -> dict:
    """GET ``url`` and return parsed JSON, retrying transient failures."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=60)
        except requests.RequestException as exc:  # network hiccup
            last_exc = exc
            _time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            return resp.json()

        # Open-Meteo returns {"error": true, "reason": "..."} with a 4xx body.
        try:
            reason = resp.json().get("reason", resp.text)
        except ValueError:
            reason = resp.text

        if resp.status_code in (429, 500, 502, 503, 504):
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {reason}")
            _time.sleep(2 ** attempt + 1)
            continue

        raise RuntimeError(f"Open-Meteo {url} -> HTTP {resp.status_code}: {reason}")

    raise RuntimeError(f"Open-Meteo {url} failed after {retries} attempts: {last_exc}")


def _hourly_frame(payload: dict, variables: list[str]) -> pd.DataFrame:
    """Turn an Open-Meteo ``hourly`` payload into a tidy DataFrame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo response missing 'hourly.time'")

    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})
    for name in variables:
        # A variable can be absent if Open-Meteo has no data for it at all.
        frame[name] = hourly.get(name, pd.NA)
    return frame


def _as_date(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


# --------------------------------------------------------------------------- #
# Per-source fetchers
# --------------------------------------------------------------------------- #
def _fetch_air_quality(
    lat: float, lon: float, start: dt.date, end: dt.date
) -> pd.DataFrame:
    payload = _get_json(
        AIR_QUALITY_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(POLLUTANT_VARS),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "UTC",
        },
    )
    return _hourly_frame(payload, POLLUTANT_VARS)


def _fetch_weather_archive(
    lat: float, lon: float, start: dt.date, end: dt.date
) -> pd.DataFrame:
    """Historical weather from ERA5. Re-clamps ``end`` if the API rejects it."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }
    try:
        payload = _get_json(WEATHER_ARCHIVE_URL, params)
    except RuntimeError as exc:
        capped = _parse_allowed_max(str(exc))
        if capped is None or capped < start:
            raise
        params["end_date"] = capped.isoformat()
        payload = _get_json(WEATHER_ARCHIVE_URL, params)
    return _hourly_frame(payload, WEATHER_VARS)


def _fetch_weather_forecast(lat: float, lon: float, past_days: int) -> pd.DataFrame:
    """Recent weather (incl. reanalysis of the last few days) to fill the gap
    the archive endpoint has not caught up to yet."""
    past_days = max(1, min(int(past_days), 92))
    payload = _get_json(
        WEATHER_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_VARS),
            "past_days": past_days,
            "forecast_days": 1,
            "timezone": "UTC",
        },
    )
    return _hourly_frame(payload, WEATHER_VARS)


def _parse_allowed_max(message: str) -> dt.date | None:
    """Extract the max allowed date from an Open-Meteo range error, e.g.
    "...out of allowed range from 1940-01-01 to 2026-08-31"."""
    matches = re.findall(r"(\d{4}-\d{2}-\d{2})", message)
    if "allowed range" in message and matches:
        return dt.date.fromisoformat(matches[-1])
    return None


# --------------------------------------------------------------------------- #
# Weather assembly (archive + forecast fill)
# --------------------------------------------------------------------------- #
def _fetch_weather(lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Weather for ``[start, end]``: archive (ERA5) as far as it reaches, then the
    forecast endpoint's ``past_days`` window to fill anything more recent."""
    today = _utc_today()
    lo = pd.Timestamp(start, tz="UTC")
    want_hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)

    parts: list[pd.DataFrame] = []
    archive_max: pd.Timestamp | None = None

    archive_end = min(end, today)
    if archive_end >= start:
        archive = _fetch_weather_archive(lat, lon, start, archive_end)
        archive = archive[archive[WEATHER_VARS].notna().any(axis=1)]
        if not archive.empty:
            parts.append(archive)
            archive_max = archive["time"].max()

    have_hi = archive_max if archive_max is not None else lo - pd.Timedelta(hours=1)
    if have_hi < want_hi:
        days_back = (today - have_hi.date()).days + 2
        parts.append(_fetch_weather_forecast(lat, lon, past_days=days_back))

    if not parts:
        return pd.DataFrame(columns=["time", *WEATHER_VARS])

    weather = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset="time", keep="first")  # archive wins over forecast
        .sort_values("time")
        .reset_index(drop=True)
    )
    return weather[(weather["time"] >= lo) & (weather["time"] <= want_hi)].reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _merge(location: str, air: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = air.merge(weather, on="time", how="left")
    merged.insert(0, "location", location)
    for col in COLUMN_ORDER:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[COLUMN_ORDER].sort_values("time").reset_index(drop=True)
    numeric = [c for c in COLUMN_ORDER if c not in ("location", "time")]
    merged[numeric] = merged[numeric].apply(pd.to_numeric, errors="coerce")
    return merged


def historical(
    location: str | dict,
    start_date: str | dt.date,
    end_date: str | dt.date,
) -> pd.DataFrame:
    """Hourly air-quality + weather for ``location`` over ``[start_date, end_date]``.

    ``location`` may be a name (``"karachi"``) or a location dict with
    ``lat``/``lon``/``name``.
    """
    loc = location if isinstance(location, dict) else get_location(location)
    start, end = _as_date(start_date), _as_date(end_date)
    if end < start:
        raise ValueError(f"end_date {end} precedes start_date {start}")

    air = _fetch_air_quality(loc["lat"], loc["lon"], start, end)
    weather = _fetch_weather(loc["lat"], loc["lon"], start, end)
    return _merge(loc["name"], air, weather)


def latest_hour(location: str | dict) -> pd.DataFrame:
    """The most recent fully-available hour for ``location`` (single-row frame).

    Weather comes from the forecast endpoint's ``past_days`` window because the
    archive endpoint lags behind the current hour.
    """
    loc = location if isinstance(location, dict) else get_location(location)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    start = (now - pd.Timedelta(days=3)).date()
    end = now.date()

    air = _fetch_air_quality(loc["lat"], loc["lon"], start, end)
    weather = _fetch_weather_forecast(loc["lat"], loc["lon"], past_days=4)
    merged = _merge(loc["name"], air, weather)

    usable = merged[(merged["time"] <= now) & merged["us_aqi"].notna()]
    if usable.empty:
        raise RuntimeError(f"No usable air-quality row for {loc['name']} at/<= {now}")
    return usable.tail(1).reset_index(drop=True)
