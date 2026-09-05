"""Data Insights - EDA charts over the historical feature store.

    streamlit run aqi_predictor/dashboard/app.py

Auto-discovered by Streamlit's native multipage nav (this file lives in
``pages/`` next to ``app.py``, so it shows up in the sidebar automatically).
Same location selector pattern as the main page (``app.py``'s ``main()``) -
no second way to pick a location. The four charts are shared with
``notebooks/eda.ipynb`` via :mod:`aqi_predictor.dashboard.eda_charts` - this
file only decides how to display them.
"""

from __future__ import annotations

import streamlit as st

from aqi_predictor.config import LOCATIONS
from aqi_predictor.dashboard import eda_charts
from aqi_predictor.feature_pipeline import store

st.set_page_config(page_title="AQI Predictor - Data Insights", page_icon="📊", layout="centered")

# Far-apart bounds so a single call pulls the whole feature store for one
# location, mirroring dataset.py's _FULL_START/_FULL_END.
_FULL_START = "2000-01-01"
_FULL_END = "2100-01-01"


@st.cache_data(ttl=1200, show_spinner="Loading historical features…")
def load_history(location_name: str):
    """Cached wrapper around the feature store read - navigating to this page
    repeatedly (or switching back and forth) doesn't re-hit Hopsworks."""
    return store.get_feature_view(_FULL_START, _FULL_END, locations=[location_name])


def main() -> None:
    st.title("Data Insights")

    names = [loc["name"] for loc in LOCATIONS]
    location = st.selectbox("Location", names, index=0, format_func=str.title)

    try:
        df = load_history(location)
    except Exception as exc:  # surface any pipeline failure, don't crash the page
        st.error(
            f"Could not load historical features for **{location}**.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )
        return

    if df.empty:
        st.warning(f"No stored features for **{location}** yet.")
        return

    st.subheader("Trend & data gaps")
    st.altair_chart(eda_charts.trend_and_gaps(df), width="stretch")

    st.subheader("Seasonality")
    st.altair_chart(eda_charts.seasonality_patterns(df), width="stretch")

    st.subheader("Category distribution")
    st.altair_chart(eda_charts.category_distribution(df), width="stretch")

    st.subheader("Feature correlation")
    st.altair_chart(eda_charts.feature_correlation(df), width="stretch")


# `streamlit run` executes each page script as "__main__" when navigated to;
# the guard just stops a plain `import` from firing a live fetch.
if __name__ == "__main__":
    main()
