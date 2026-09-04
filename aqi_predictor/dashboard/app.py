"""Streamlit dashboard for the live AQI forecast.

    streamlit run aqi_predictor/dashboard/app.py

Reads from the real pipeline (:func:`aqi_predictor.inference_pipeline.predict.forecast`)
- no mock data. Deployment to Streamlit Community Cloud is a manual step done on
streamlit.io after the repo is pushed (see PROGRESS.md).
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from aqi_predictor.aqi_scale import aqi_category
from aqi_predictor.config import LOCATIONS
from aqi_predictor.inference_pipeline import predict

# hopsworks / hsfs stream tqdm progress bars to stdout; Streamlit captures stdout
# per script run and the volume can stall the render. tqdm reads this on each bar
# creation, so setting it here (before the first forecast call) is enough.
os.environ.setdefault("TQDM_DISABLE", "1")

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")

# Lower bound of each US-EPA AQI category (the last band, 300+, is open-ended and
# clipped to the chart's y-max at draw time).
_BAND_LOWERS = [0, 50, 100, 150, 200, 300]

_CARD_CSS = """
<style>
.aqi-card {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 14px;
    padding: 0.9rem 1rem 0.85rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08), 0 6px 18px rgba(0, 0, 0, 0.06);
    height: 100%;
}
.aqi-card .aqi-card-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    opacity: 0.6;
}
.aqi-card .aqi-card-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 0.15rem 0 0.35rem;
}
.aqi-card .aqi-card-pill {
    display: inline-block;
    padding: 0.12rem 0.6rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
}
.aqi-card .aqi-card-time {
    font-size: 0.7rem;
    opacity: 0.5;
    margin-top: 0.5rem;
}
</style>
"""


@st.cache_data(ttl=1200, show_spinner="Fetching live forecast…")
def load_forecast(location_name: str) -> dict:
    """Cached wrapper around the live pipeline.

    ``location_name`` is an explicit argument, so ``st.cache_data`` keys on it:
    switching cities returns that city's own (fresh-or-cached) result, never a
    stale entry for the previously selected city.
    """
    return predict.forecast(location_name)


def _text_on(hex_colour: str) -> str:
    """Readable text colour (near-black / white) for a solid ``hex_colour`` fill."""
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1a1a1a" if luminance > 140 else "#ffffff"


def _stat_card(label: str, value: float, timestamp: str) -> str:
    """HTML for one top-row stat card: big AQI number, category pill, timestamp."""
    cat_label, colour = aqi_category(value)
    when = pd.to_datetime(timestamp).strftime("%b %d, %H:%M UTC")
    return (
        '<div class="aqi-card">'
        f'<div class="aqi-card-label">{label}</div>'
        f'<div class="aqi-card-value">{value:.0f}</div>'
        f'<div class="aqi-card-pill" style="background:{colour};color:{_text_on(colour)}">'
        f"{cat_label}</div>"
        f'<div class="aqi-card-time">{when}</div>'
        "</div>"
    )


def _category_bands(y_max: float) -> pd.DataFrame:
    """One row per visible EPA category band: ``y`` / ``y2`` range + its hex colour."""
    rows: list[dict] = []
    for i, lower in enumerate(_BAND_LOWERS):
        if lower >= y_max:
            break
        nominal_upper = _BAND_LOWERS[i + 1] if i + 1 < len(_BAND_LOWERS) else y_max
        mid = lower + (nominal_upper - lower) / 2 if i + 1 < len(_BAND_LOWERS) else lower + 1
        label, colour = aqi_category(mid)
        rows.append(
            {
                "y": lower,
                "y2": min(nominal_upper, y_max),
                "label": label,
                "colour": colour,
            }
        )
    return pd.DataFrame(rows)


def _trend_chart(recent: pd.DataFrame) -> alt.LayerChart:
    """48h observed ``us_aqi`` line over translucent EPA category bands."""
    # Vega-Lite mis-parses tz-aware ISO strings (drops every row -> blank chart);
    # the data is all UTC, so hand it naive timestamps.
    recent = recent.assign(
        time=pd.to_datetime(recent["time"], utc=True).dt.tz_localize(None)
    )
    y_max = max(float(recent["us_aqi"].max()), 55.0) + 12.0
    y_scale = alt.Scale(domain=[0, y_max], nice=False)
    x_min, x_max = recent["time"].min(), recent["time"].max()

    bands = _category_bands(y_max)
    # Give every band an explicit x-span (the observed window) so the rects fill
    # the plot width, rather than relying on a rect-with-no-x spanning the layer.
    bands["x"] = x_min
    bands["x2"] = x_max
    band_layer = (
        alt.Chart(bands)
        .mark_rect(opacity=0.22)
        .encode(
            x=alt.X("x:T", title=None),
            x2="x2:T",
            y=alt.Y("y:Q", scale=y_scale, title="US AQI"),
            y2="y2:Q",
            color=alt.Color("colour:N", scale=None, legend=None),
        )
    )
    # A dark "halo" line under a thinner white line: on the light theme the dark
    # one reads, on the dark theme the white one does - so the trend is visible
    # over any band colour either way.
    line_enc = alt.Chart(recent).encode(
        x=alt.X("time:T", title=None),
        y=alt.Y("us_aqi:Q", scale=y_scale, title="US AQI"),
    )
    halo = line_enc.mark_line(color="#1a1a1a", strokeWidth=4)
    line = line_enc.mark_line(color="#f5f5f5", strokeWidth=1.8).encode(
        tooltip=[
            alt.Tooltip("time:T", title="time (UTC)"),
            alt.Tooltip("us_aqi:Q", title="US AQI", format=".0f"),
        ],
    )
    return (
        alt.layer(band_layer, halo, line)
        .properties(height=280)
        .configure_view(stroke=None)
    )


def _forecast_chart(fc: pd.DataFrame) -> alt.LayerChart:
    """One bar per horizon, each coloured by its own predicted category."""
    fc = fc.assign(
        target_time=pd.to_datetime(fc["target_time"], utc=True).dt.tz_localize(None)
    )
    order = ["+1h", "+24h", "+48h", "+72h"]
    cat_colour = {row.category: row.colour for row in fc.itertuples()}
    colour_scale = alt.Scale(
        domain=list(cat_colour), range=list(cat_colour.values())
    )
    base = alt.Chart(fc).encode(
        x=alt.X(
            "horizon_label:N",
            sort=order,
            title=None,
            scale=alt.Scale(paddingInner=0.35, paddingOuter=0.25),
        ),
        y=alt.Y("predicted_us_aqi:Q", title="US AQI", scale=alt.Scale(domainMin=0)),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        color=alt.Color("category:N", scale=colour_scale, legend=None),
        tooltip=[
            alt.Tooltip("horizon_label:N", title="horizon"),
            alt.Tooltip("target_time:T", title="target time (UTC)"),
            alt.Tooltip("predicted_us_aqi:Q", title="US AQI", format=".0f"),
            alt.Tooltip("category:N", title="category"),
        ],
    )
    labels = base.mark_text(
        dy=-7, fontWeight="bold", color="#f5f5f5", stroke="#1a1a1a", strokeWidth=0.5
    ).encode(text=alt.Text("predicted_us_aqi:Q", format=".0f"))
    return alt.layer(bars, labels).properties(height=260).configure_view(stroke=None)


def _render(result: dict) -> None:
    cur = result["current"]
    by_horizon = {f["horizon_hours"]: f for f in result["forecasts"]}

    st.caption(
        f"Current reading {cur['time']} · forecast generated {result['generated_at']}"
    )

    cards: list[tuple[str, float, str]] = [("Now", cur["us_aqi"], cur["time"])]
    for horizon in (24, 48, 72):
        f = by_horizon.get(horizon)
        if f is not None:
            cards.append((f"+{horizon}h", f["predicted_us_aqi"], f["target_time"]))

    for column, (label, value, timestamp) in zip(st.columns(len(cards)), cards):
        column.markdown(_stat_card(label, value, timestamp), unsafe_allow_html=True)

    recent = pd.DataFrame(result.get("recent", []))
    if not recent.empty:
        recent["time"] = pd.to_datetime(recent["time"])
        st.subheader("Last 48 hours (observed)")
        st.altair_chart(_trend_chart(recent), width="stretch")

    fc = pd.DataFrame(result["forecasts"])
    fc["target_time"] = pd.to_datetime(fc["target_time"])
    fc["category"] = fc["predicted_us_aqi"].map(lambda v: aqi_category(v)[0])
    fc["colour"] = fc["predicted_us_aqi"].map(lambda v: aqi_category(v)[1])
    fc["model"] = fc["model_name"] + " v" + fc["model_version"].astype(str)
    fc["horizon_label"] = "+" + fc["horizon_hours"].astype(str) + "h"

    st.subheader("Forecast (+1h / +24h / +48h / +72h)")
    st.altair_chart(_forecast_chart(fc), width="stretch")

    with st.expander("Forecast details", expanded=False):
        st.dataframe(
            fc[["horizon_hours", "target_time", "predicted_us_aqi", "category", "model"]],
            hide_index=True,
            column_config={
                "horizon_hours": "horizon (h)",
                "target_time": "target time (UTC)",
                "predicted_us_aqi": "predicted US AQI",
            },
        )


def main() -> None:
    st.title("🌫️ AQI Predictor")
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    names = [loc["name"] for loc in LOCATIONS]
    location = st.selectbox("Location", names, index=0, format_func=str.title)

    try:
        result = load_forecast(location)
    except Exception as exc:  # surface any pipeline failure, don't crash the page
        st.error(
            f"Could not load the forecast for **{location}**.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )
        return

    _render(result)


# `streamlit run app.py` and AppTest both execute this file as "__main__"; the
# guard just stops a plain `import` from firing a live fetch.
if __name__ == "__main__":
    main()
