"""US AQI category + colour lookup.

Standard US EPA AQI breakpoints (thresholds unchanged). Shared across the
project (the dashboard, and hazardous-AQI alerting) - kept dependency-free on
purpose, no Streamlit / pandas import here.

The colours are a muted pastel-to-saturated palette (not the vivid official
AirNow one) that visibly escalates toward the top of the scale: soft and
calm for Good/Moderate, progressively more saturated through Unhealthy for
Sensitive Groups / Unhealthy / Very Unhealthy, darkest and most saturated at
Hazardous - so worsening air quality reads as increasingly urgent rather than
all six categories looking equally soft. The dashboard's hazard banner
(``app.py``'s ``.hazard-banner``) is deliberately independent of this palette
- it stays a fixed, bold, saturated red regardless of this module's colours,
since it's a safety-critical element that must never look "on theme."
"""

from __future__ import annotations

import math

# (inclusive upper bound, label, hex colour) - muted pastel-to-saturated scale.
_BREAKPOINTS: list[tuple[float, str, str]] = [
    (50, "Good", "#B7E4C7"),  # soft mint
    (100, "Moderate", "#F0E6A6"),  # soft sand / muted yellow
    (150, "Unhealthy for Sensitive Groups", "#F3C393"),  # soft peach
    (200, "Unhealthy", "#D98C87"),  # muted dusty rose
    (300, "Very Unhealthy", "#A15C7A"),  # deeper rose-mauve, more saturated
]
_HAZARDOUS: tuple[str, str] = ("Hazardous", "#5C1A2E")  # deep muted maroon, darkest/most saturated

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
