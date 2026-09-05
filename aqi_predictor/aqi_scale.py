"""US AQI category + colour lookup.

Standard US EPA AQI breakpoints. Shared across the project (the dashboard, and a
later hazardous-AQI alerting phase) - kept dependency-free on purpose, no
Streamlit / pandas import here.
"""

from __future__ import annotations

import math

# (inclusive upper bound, label, hex colour) - official AirNow palette.
_BREAKPOINTS: list[tuple[float, str, str]] = [
    (50, "Good", "#00e400"),
    (100, "Moderate", "#ffff00"),
    (150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (200, "Unhealthy", "#ff0000"),
    (300, "Very Unhealthy", "#8f3f97"),
]
_HAZARDOUS: tuple[str, str] = ("Hazardous", "#7e0023")

# Every category's (label, colour), worst-to-best-independent order (Good ->
# Hazardous) - derived from the same breakpoint data ``aqi_category`` uses, so
# callers needing the full category list (e.g. a distribution chart) don't
# hardcode a second copy of the labels/colours.
CATEGORIES: list[tuple[str, str]] = [(label, colour) for _upper, label, colour in _BREAKPOINTS] + [
    _HAZARDOUS
]


def aqi_category(value: float) -> tuple[str, str]:
    """Return ``(label, colour_hex)`` for a US AQI ``value``.

    Boundaries are inclusive on the lower category (e.g. 50 -> Good, 51 ->
    Moderate). Anything above 300 is Hazardous. Raises ``ValueError`` for a
    non-finite value.
    """
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"AQI value must be finite, got {value!r}")
    for upper, label, colour in _BREAKPOINTS:
        if v <= upper:
            return label, colour
    return _HAZARDOUS


def hazardous_horizons(current: dict, forecasts: list[dict]) -> list[str]:
    """Which of ``current`` / each forecast entry is in the "Hazardous"
    category, as human-readable labels (``"now"``, ``"+24h"``, ...).

    Takes the same ``current`` / ``forecasts`` dicts
    ``predict.forecast()`` already returns - no new fetch needed. Empty list
    if nothing is hazardous.
    """
    hazardous: list[str] = []
    if aqi_category(current["us_aqi"])[0] == "Hazardous":
        hazardous.append("now")
    for fc in forecasts:
        if aqi_category(fc["predicted_us_aqi"])[0] == "Hazardous":
            hazardous.append(f"+{fc['horizon_hours']}h")
    return hazardous
