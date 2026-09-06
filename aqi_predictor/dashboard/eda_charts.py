"""Shared EDA chart functions - used by both ``notebooks/eda.ipynb`` and the
dashboard's "Data Insights" page, so the analysis logic lives in exactly one
place and each caller only decides how to *display* it.

Every function takes a single-location engineered feature frame (the shape
``store.get_feature_view(start, end, locations=[name])`` returns, or an
equivalent frame with ``time``/``us_aqi``/pollutant/weather columns) and
returns an Altair chart object.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from aqi_predictor.aqi_scale import CATEGORIES, aqi_category
from aqi_predictor.config import POLLUTANT_VARS, WEATHER_VARS

# Altair's default data transformer refuses more than 5000 rows (it's meant
# to catch accidentally embedding huge datasets); the full feature-store
# history for even one location is already well past that. This is the
# built-in escape hatch - no new dependency (e.g. vegafusion) needed.
alt.data_transformers.disable_max_rows()

_CATEGORY_ORDER = [label for label, _colour in CATEGORIES]
_CATEGORY_COLOUR_SCALE = alt.Scale(
    domain=_CATEGORY_ORDER, range=[colour for _label, colour in CATEGORIES]
)


def _missing_hour_spans(df: pd.DataFrame) -> pd.DataFrame:
    """Start/end timestamps of every gap in ``df``'s hourly ``time`` column."""
    times = pd.to_datetime(df["time"])
    full_idx = pd.date_range(times.min(), times.max(), freq="h", tz=times.dt.tz)
    present = pd.Series(True, index=times).reindex(full_idx, fill_value=False)
    missing = ~present

    if not missing.any():
        return pd.DataFrame(columns=["start", "end"])

    run_id = (missing != missing.shift()).cumsum()
    spans = []
    for idx in missing[missing].groupby(run_id[missing]).groups.values():
        spans.append({"start": idx.min(), "end": idx.max() + pd.Timedelta(hours=1)})
    return pd.DataFrame(spans)


# Above this many hourly points, the line is resampled to a daily mean before
# charting - a couple of years of hourly history renders as ~500-1000 points
# either way, and embedding every raw hour balloons the Vega-Lite spec (an
# 18k-row history pushed one chart past 29MB). Gap detection below still runs
# on the full, un-resampled hourly index, so a gap doesn't need to span a
# whole day to still show up as a shaded band.
_MAX_TREND_POINTS = 2000


def trend_and_gaps(df: pd.DataFrame) -> alt.Chart:
    """``us_aqi`` over time, with any missing-hour gaps shaded red underneath."""
    d = df.sort_values("time")
    spans = _missing_hour_spans(d)

    line_source = d[["time", "us_aqi"]]
    if len(line_source) > _MAX_TREND_POINTS:
        line_source = (
            line_source.set_index("time")["us_aqi"].resample("1D").mean().reset_index()
        )

    line = (
        alt.Chart(line_source)
        .mark_line(color="#2563eb")
        .encode(
            x=alt.X("time:T", title=None),
            y=alt.Y("us_aqi:Q", title="US AQI"),
            tooltip=[alt.Tooltip("time:T", title="time (UTC)"), alt.Tooltip("us_aqi:Q", format=".0f")],
        )
    )

    if spans.empty:
        return line.properties(height=280).configure_view(stroke=None)

    gaps = (
        alt.Chart(spans)
        .mark_rect(opacity=0.25, color="#ef4444")
        .encode(x=alt.X("start:T"), x2="end:T")
    )
    return alt.layer(gaps, line).properties(height=280).configure_view(stroke=None)


def seasonality_patterns(df: pd.DataFrame) -> alt.Chart:
    """Mean ``us_aqi`` by hour-of-day and by month-of-year, side by side."""
    d = df.copy()
    times = pd.to_datetime(d["time"])
    if "hour" not in d.columns:
        d["hour"] = times.dt.hour
    if "month" not in d.columns:
        d["month"] = times.dt.month

    by_hour = d.groupby("hour", as_index=False)["us_aqi"].mean()
    by_month = d.groupby("month", as_index=False)["us_aqi"].mean()

    hour_chart = (
        alt.Chart(by_hour)
        .mark_line(point=True, color="#2563eb")
        .encode(
            x=alt.X("hour:O", title="hour of day (UTC)"),
            y=alt.Y("us_aqi:Q", title="mean US AQI"),
            tooltip=["hour", alt.Tooltip("us_aqi:Q", format=".1f")],
        )
        .properties(height=260, title="By hour of day")
    )
    month_chart = (
        alt.Chart(by_month)
        .mark_bar(color="#6b7280")
        .encode(
            x=alt.X("month:O", title="month"),
            y=alt.Y("us_aqi:Q", title="mean US AQI"),
            tooltip=["month", alt.Tooltip("us_aqi:Q", format=".1f")],
        )
        .properties(height=260, title="By month of year")
    )
    return alt.hconcat(hour_chart, month_chart).configure_view(stroke=None)


def category_distribution(df: pd.DataFrame) -> alt.Chart:
    """Count of hours spent in each US AQI category, coloured per ``aqi_scale``."""
    labels = df["us_aqi"].map(lambda v: aqi_category(v)[0])
    counts = labels.value_counts().reindex(_CATEGORY_ORDER, fill_value=0)
    counts_df = counts.rename("count").rename_axis("category").reset_index()

    return (
        alt.Chart(counts_df)
        .mark_bar()
        .encode(
            x=alt.X("category:N", sort=_CATEGORY_ORDER, title=None),
            y=alt.Y("count:Q", title="hours"),
            color=alt.Color("category:N", scale=_CATEGORY_COLOUR_SCALE, legend=None),
            tooltip=["category:N", "count:Q"],
        )
        .properties(height=280)
        .configure_view(stroke=None)
        .configure_axisX(labelAngle=-30)
    )


def feature_correlation(df: pd.DataFrame) -> alt.Chart:
    """Correlation heatmap among the core pollutant/weather columns (incl. ``us_aqi``)."""
    cols = [c for c in [*POLLUTANT_VARS, *WEATHER_VARS] if c in df.columns]
    corr = df[cols].corr()
    long = corr.stack().rename("corr").rename_axis(["row", "col"]).reset_index()

    return (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("col:N", title=None, sort=cols),
            y=alt.Y("row:N", title=None, sort=cols),
            color=alt.Color(
                "corr:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="correlation",
            ),
            tooltip=[
                alt.Tooltip("row:N"),
                alt.Tooltip("col:N"),
                alt.Tooltip("corr:Q", format=".2f"),
            ],
        )
        .properties(height=320)
        .configure_view(stroke=None)
    )
