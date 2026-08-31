"""Shared configuration: locations, paths, constants, and environment loading.

Environment variables (secrets only) are read from a ``.env`` file at the repo
root via ``python-dotenv``. Non-secret settings such as coordinates live here in
code, not in ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                     # raw merged API pulls
FEATURE_STORE_DIR = DATA_DIR / "feature_store"  # local Hopsworks fallback

for _d in (DATA_DIR, RAW_DIR, FEATURE_STORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Environment (secrets)
# --------------------------------------------------------------------------- #
load_dotenv(PROJECT_ROOT / ".env")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "")

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
# Multi-location schema from day one: every feature row carries a location key.
# Only Karachi is active right now; add dicts here to onboard more cities.
LOCATIONS: list[dict] = [
    {"name": "karachi", "lat": 24.8608, "lon": 67.0104},
]


def get_location(name: str) -> dict:
    """Return the location dict for ``name`` (case-insensitive)."""
    key = name.strip().lower()
    for loc in LOCATIONS:
        if loc["name"] == key:
            return loc
    raise KeyError(f"Unknown location {name!r}; known: {[l['name'] for l in LOCATIONS]}")


# --------------------------------------------------------------------------- #
# Data / backfill constants
# --------------------------------------------------------------------------- #
# Start of the usable Open-Meteo air-quality history we care about.
BACKFILL_START_DATE = "2024-07-29"

# Open-Meteo endpoints
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Hourly variables
POLLUTANT_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
]

# Columns that must be present for a feature row to be usable.
REQUIRED_COLUMNS = ["us_aqi", "pm2_5", "pm10"]
