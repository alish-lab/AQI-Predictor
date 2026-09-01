"""Streamlit dashboard for the live AQI forecast.

    streamlit run aqi_predictor/dashboard/app.py

Reads from the real pipeline (:func:`aqi_predictor.inference_pipeline.predict.forecast`)
- no mock data. Deployment to Streamlit Community Cloud is a manual step done on
streamlit.io after the repo is pushed (see PROGRESS.md).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from aqi_predictor.aqi_scale import aqi_category
from aqi_predictor.config import LOCATIONS
from aqi_predictor.inference_pipeline import predict

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")


@st.cache_data(ttl=1200, show_spinner="Fetching live forecast…")
def load_forecast(location_name: str) -> dict:
    """Cached wrapper around the live pipeline.

    ``location_name`` is an explicit argument, so ``st.cache_data`` keys on it:
    switching cities returns that city's own (fresh-or-cached) result, never a
    stale entry for the previously selected city.
    """
    return predict.forecast(location_name)


def _aqi_badge(value: float) -> str:
    label, colour = aqi_category(value)
    return (
        f'<div style="padding:0.6rem 1rem;border-radius:0.5rem;background:{colour};'
        f'color:#111;font-weight:700;display:inline-block;font-size:1.1rem;">'
        f"US AQI {value:.0f} — {label}</div>"
    )


def _render(result: dict) -> None:
    cur = result["current"]
    st.markdown(_aqi_badge(cur["us_aqi"]), unsafe_allow_html=True)
    st.caption(
        f"Current reading {cur['time']} · forecast generated {result['generated_at']}"
    )

    recent = pd.DataFrame(result.get("recent", []))
    if not recent.empty:
        recent["time"] = pd.to_datetime(recent["time"])
        st.subheader("Last 48 hours (observed)")
        st.line_chart(recent.set_index("time")["us_aqi"], y_label="US AQI")

    fc = pd.DataFrame(result["forecasts"])
    fc["target_time"] = pd.to_datetime(fc["target_time"])
    fc["category"] = fc["predicted_us_aqi"].map(lambda v: aqi_category(v)[0])
    fc["model"] = fc["model_name"] + " v" + fc["model_version"].astype(str)

    st.subheader("Forecast (+1h / +24h / +48h / +72h)")
    st.line_chart(
        fc.set_index("target_time")["predicted_us_aqi"], y_label="US AQI"
    )
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
