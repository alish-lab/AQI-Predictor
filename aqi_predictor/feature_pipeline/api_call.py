from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import urllib.request
import urllib.parse

import pandas as pd

# repo root is three levels up: aqi_predictor/feature_pipeline/api_call.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OUTPUT_FILE = PROJECT_ROOT / "data" / "raw_aqi.csv"
RAW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_HOURLY = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def _build_request_url(latitude: float, longitude: float, hourly: Iterable[str], start_date: str, end_date: str, timezone: str = "auto") -> str:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }
    return f"{OPEN_METEO_AQI_URL}?{urllib.parse.urlencode(params)}"


def fetch_air_quality(
    latitude: float = 24.8608,
    longitude: float = 67.0104,
    start_date: str = "2024-07-29",
    end_date: str = "2026-08-10",
    hourly_variables: Iterable[str] | None = None,
    timezone: str = "auto",
) -> pd.DataFrame:
    hourly_variables = list(hourly_variables or DEFAULT_HOURLY)
    url = _build_request_url(latitude, longitude, hourly_variables, start_date, end_date, timezone)

    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Open-Meteo API returned HTTP {response.status}")
        payload = json.load(response)

    hourly_data = payload.get("hourly")
    if not hourly_data or "time" not in hourly_data:
        raise RuntimeError("Open-Meteo response missing hourly time data")

    df = pd.DataFrame({"time": pd.to_datetime(hourly_data["time"], utc=True)})
    for name in hourly_variables:
        if name not in hourly_data:
            raise RuntimeError(f"Hourly variable '{name}' not returned by API")
        df[name] = hourly_data[name]

    return df


def get_hourly_dataframe(**kwargs) -> pd.DataFrame:
    df = fetch_air_quality(**kwargs)
    return df


def main() -> None:
    df = get_hourly_dataframe()
    df.to_csv(RAW_OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {RAW_OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()
