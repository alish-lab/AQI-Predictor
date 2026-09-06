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

from aqi_predictor.aqi_scale import CATEGORIES, aqi_category, hazardous_horizons
from aqi_predictor.config import LOCATIONS
from aqi_predictor.dashboard import theme
from aqi_predictor.feature_pipeline import store
from aqi_predictor.inference_pipeline import predict
from aqi_predictor.training_pipeline import registry

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
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.45);
    border-radius: 16px;
    padding: 0.9rem 1rem 0.85rem;
    box-shadow: 0 4px 24px rgba(31, 41, 55, 0.12), 0 1px 3px rgba(31, 41, 55, 0.06);
    height: 100%;
    /* the glass card is always light by design, so its text must always be
       dark regardless of the viewer's Streamlit theme (light or dark) - don't
       let it inherit ambient theme text color. */
    color: #1a1a1a;
}
.aqi-card .aqi-card-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    opacity: 0.65;
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
.legend-pill {
    display: block;
    padding: 0.3rem 0.6rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.pollutant-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    padding: 0.2rem 0;
    border-bottom: 1px solid rgba(128, 128, 128, 0.15);
}
/* Deliberately NOT part of the pastel theme above - a safety-critical
   element stays bold/saturated red regardless of the palette in aqi_scale.py. */
.hazard-banner {
    background: #7e0023;
    color: #ffffff;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.1rem;
    font-weight: 700;
    font-size: 0.95rem;
    box-shadow: 0 2px 10px rgba(126, 0, 35, 0.4);
}
</style>
"""


# Decorative, category-appropriate weather-style glyphs next to each stat
# card's label - plain unicode, no new asset/icon-font dependency. Purely a
# display touch; falls back to no icon for an unrecognised label.
_CATEGORY_ICON: dict[str, str] = {
    "Good": "☀️",
    "Moderate": "🌤️",
    "Unhealthy for Sensitive Groups": "⛅",
    "Unhealthy": "☁️",
    "Very Unhealthy": "🌁",
    "Hazardous": "🌫️",
}

_POLLUTANT_LABELS: dict[str, str] = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone",
}


def _hazard_banner_html(horizons: list[str]) -> str:
    """Prominent top-of-page banner naming which horizon(s) triggered it.

    ``horizons`` is whatever :func:`aqi_scale.hazardous_horizons` returns
    (e.g. ``["now", "+24h"]``) - always non-empty when this is called.
    """
    where = ", ".join(horizons)
    return (
        '<div class="hazard-banner">'
        f"HAZARDOUS AIR QUALITY (US AQI &gt; 300) predicted at: {where}. "
        "Limit outdoor exposure."
        "</div>"
    )


@st.cache_data(ttl=1200, show_spinner="Fetching live forecast…")
def load_forecast(location_name: str) -> dict:
    """Cached wrapper around the live pipeline.

    ``location_name`` is an explicit argument, so ``st.cache_data`` keys on it:
    switching cities returns that city's own (fresh-or-cached) result, never a
    stale entry for the previously selected city.
    """
    return predict.forecast(location_name)


@st.cache_data(ttl=1200, show_spinner=False)
def load_latest_pollutants(location_name: str) -> dict[str, float] | None:
    """Latest stored pollutant readings for the sidebar breakdown.

    A small, separate read via the same ``store.get_feature_view`` the Data
    Insights page already uses - ``predict.forecast()``'s own return dict
    isn't touched/extended for this, keeping the serving pipeline untouched.
    ``None`` if nothing is stored yet for this location.
    """
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=2)).date().isoformat()
    end = now.date().isoformat()
    df = store.get_feature_view(start, end, locations=[location_name])
    if df.empty:
        return None
    latest = df.sort_values("time").iloc[-1]
    return {
        col: float(latest[col])
        for col in _POLLUTANT_LABELS
        if col in latest.index and pd.notna(latest[col])
    }


@st.cache_data(ttl=1200, show_spinner=False)
def load_model_performance(names: tuple[str, ...]) -> list[dict]:
    """Cached wrapper around ``registry.current_model_metrics`` - a one-time
    snapshot of whatever's currently served, not a historical trend. No new
    registry queries beyond one ``load_best_model`` per name."""
    return registry.current_model_metrics(list(names))


def _text_on(hex_colour: str) -> str:
    """Readable text colour (near-black / white) for a solid ``hex_colour`` fill."""
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1a1a1a" if luminance > 140 else "#ffffff"


def _stat_card(label: str, value: float, timestamp: str) -> str:
    """HTML for one top-row stat card: big AQI number, category pill, timestamp."""
    cat_label, colour = aqi_category(value)
    icon = _CATEGORY_ICON.get(cat_label, "")
    when = pd.to_datetime(timestamp).strftime("%b %d, %H:%M UTC")
    return (
        '<div class="aqi-card">'
        f'<div class="aqi-card-label">{icon} {label}</div>'
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
        .mark_rect(opacity=0.13)
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
    # interpolate="monotone" - a genuinely smooth curve is honest here: this is
    # real hourly-resolution observed history, not sparse forecast points.
    halo = line_enc.mark_line(color="#3b4a5a", strokeWidth=2.8, interpolate="monotone")
    line = line_enc.mark_line(color="#f5f5f5", strokeWidth=1.8, interpolate="monotone").encode(
        tooltip=[
            alt.Tooltip("time:T", title="time (UTC)"),
            alt.Tooltip("us_aqi:Q", title="US AQI", format=".0f"),
        ],
    )
    return (
        alt.layer(band_layer, halo, line)
        .properties(height=270, background="transparent")
        .configure_view(stroke=None, fill=None)
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

    # Mark whichever of the 4 (sparse, not hourly) forecast points is highest -
    # still bars, still visually honest about being 4 discrete predictions.
    peak_row = fc.loc[[fc["predicted_us_aqi"].idxmax()]]
    peak_text = (
        alt.Chart(peak_row)
        .mark_text(dy=-24, fontWeight="bold", fontSize=11, color="#5C1A2E")
        .encode(
            x=alt.X("horizon_label:N", sort=order),
            y=alt.Y("predicted_us_aqi:Q"),
            text=alt.value("Predicted Peak"),
        )
    )
    return alt.layer(bars, labels, peak_text).properties(height=270, background="transparent").configure_view(stroke=None, fill=None)


_SHAP_UP_COLOUR = "#e2434d"   # pushes predicted AQI up (worse air)
_SHAP_DOWN_COLOUR = "#3b82f6"  # pushes predicted AQI down (better air)


def _local_shap_chart(top_features: list[dict]) -> alt.Chart:
    """Horizontal bar chart of one forecast's top-5 local SHAP features."""
    df = pd.DataFrame(top_features)
    df["direction"] = df["shap_value"].map(
        lambda v: "pushes AQI up" if v > 0 else "pushes AQI down"
    )
    order = df.assign(_abs=df["shap_value"].abs()).sort_values("_abs")["feature"].tolist()
    colour_scale = alt.Scale(
        domain=["pushes AQI up", "pushes AQI down"],
        range=[_SHAP_UP_COLOUR, _SHAP_DOWN_COLOUR],
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort=order, title=None),
            x=alt.X("shap_value:Q", title="SHAP value (impact on predicted AQI)"),
            color=alt.Color("direction:N", scale=colour_scale, legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("feature:N", title="feature"),
                alt.Tooltip("value:Q", title="input value", format=".2f"),
                alt.Tooltip("shap_value:Q", title="SHAP value", format="+.2f"),
            ],
        )
        .properties(height=220, background="transparent")
        .configure_view(stroke=None, fill=None)
    )


def _global_shap_chart(shap_importance: dict[str, float], top_n: int = 10) -> alt.Chart:
    """Horizontal bar chart of a model's global mean(|SHAP|) feature importance."""
    ranked = sorted(shap_importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    df = pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"])
    order = df.sort_values("mean_abs_shap")["feature"].tolist()
    return (
        alt.Chart(df)
        # muted slate from the restyle (matches the trend chart's halo), not
        # the old flat gray left over from before the theme existed.
        .mark_bar(color="#3b4a5a")
        .encode(
            y=alt.Y("feature:N", sort=order, title=None),
            x=alt.X("mean_abs_shap:Q", title="mean |SHAP value|"),
            tooltip=[
                alt.Tooltip("feature:N", title="feature"),
                alt.Tooltip("mean_abs_shap:Q", title="mean |SHAP|", format=".3f"),
            ],
        )
        .properties(height=220, background="transparent")
        .configure_view(stroke=None, fill=None)
    )


def _render_explain_section(fc: pd.DataFrame) -> None:
    """'Explain this forecast' section: local SHAP bars + a global importance chart.

    Picked via a horizon selector rather than one chart per card, so this stays
    out of the top stat-card row entirely.
    """
    st.subheader("Explain this forecast")
    horizon_labels = list(fc["horizon_label"])
    chosen_label = st.selectbox("Horizon", horizon_labels, index=0, key="shap_horizon")
    chosen = fc.loc[fc["horizon_label"] == chosen_label].iloc[0]

    col_local, col_global = st.columns(2)
    with col_local:
        st.markdown(f"**Top features - {chosen_label} local impact**")
        top_features = chosen.get("top_features")
        if top_features:
            st.altair_chart(_local_shap_chart(top_features), width="stretch")
        else:
            st.caption(
                f"No SHAP explanation available for {chosen['model']} (non-tree model)."
            )
    with col_global:
        st.markdown(f"**Global feature importance - {chosen['model_name']}**")
        shap_importance = chosen.get("shap_importance")
        if shap_importance:
            st.altair_chart(_global_shap_chart(shap_importance), width="stretch")
        else:
            st.caption(f"No global SHAP importance available for {chosen['model_name']}.")


def _render_model_performance(fc: pd.DataFrame) -> None:
    """Plain table: one row per forecast horizon, showing the currently
    served model + its test RMSE/MAE/R2 - a one-time snapshot pulled from the
    registry (``registry.current_model_metrics``, one ``load_best_model`` per
    horizon), not a historical trend. No glass-card treatment needed for a
    plain data table."""
    names = tuple(fc["model_name"])
    by_name = {m["name"]: m for m in load_model_performance(names)}

    rows = []
    for _, row in fc.iterrows():
        m = by_name.get(row["model_name"])
        if m is None:
            continue
        rows.append(
            {
                "horizon": row["horizon_label"],
                "model": f"{m['algorithm']} v{m['version']}",
                "RMSE": m["rmse"],
                "MAE": m["mae"],
                "R2": m["r2"],
            }
        )
    if not rows:
        return

    st.subheader("Model Performance")
    st.caption("Currently served model per horizon - a snapshot, not a historical trend.")
    st.dataframe(pd.DataFrame(rows), hide_index=True)


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

    _render_model_performance(fc)
    _render_explain_section(fc)


def _render_sidebar(location: str) -> None:
    """AQI scale legend (new palette) + a key-pollutants breakdown for the
    selected location - reuses ``store.get_feature_view`` read-only, no
    change to the serving pipeline."""
    with st.sidebar:
        st.markdown("### AQI Scale")
        for label, colour in CATEGORIES:
            st.markdown(
                f'<div class="legend-pill" style="background:{colour};'
                f'color:{_text_on(colour)}">{label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Key Pollutants")
        try:
            pollutants = load_latest_pollutants(location)
        except Exception:
            # store.get_feature_view()'s only offline read path (Hopsworks'
            # Arrow Flight Query Service) depends on live network reachability
            # with no working fallback in the installed hsfs client - a
            # transient failure here shouldn't take down the whole dashboard.
            # See the comment above store.get_feature_view() for the full
            # explanation.
            pollutants = None
            st.caption("Pollutant breakdown unavailable right now.")
        else:
            if pollutants:
                for col, val in pollutants.items():
                    st.markdown(
                        f'<div class="pollutant-row"><span>{_POLLUTANT_LABELS[col]}</span>'
                        f"<span>{val:.1f}</span></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No recent pollutant data.")


def main() -> None:
    st.title("🌫️ AQI Predictor")
    st.markdown(theme.BACKGROUND_CSS, unsafe_allow_html=True)
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    names = [loc["name"] for loc in LOCATIONS]
    location = st.selectbox("Location", names, index=0, format_func=str.title)
    _render_sidebar(location)

    try:
        result = load_forecast(location)
    except Exception as exc:  # surface any pipeline failure, don't crash the page
        st.error(
            f"Could not load the forecast for **{location}**.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )
        return

    hazardous = hazardous_horizons(result["current"], result["forecasts"])
    if hazardous:
        st.markdown(_hazard_banner_html(hazardous), unsafe_allow_html=True)

    _render(result)


# `streamlit run app.py` and AppTest both execute this file as "__main__"; the
# guard just stops a plain `import` from firing a live fetch.
if __name__ == "__main__":
    main()
