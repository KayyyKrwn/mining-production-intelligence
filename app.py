"""Mining Production & Predictive Maintenance Intelligence — app entrypoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.dashboard_data import load_data
from src.ui import render_sidebar

st.set_page_config(
    page_title="Mining Production Intelligence",
    page_icon="⛏️",
    layout="wide",
)

st.title("⛏️ Mining Production & Predictive Maintenance Intelligence")
st.markdown(
    "Executive analytics for open-pit mining operations — production, equipment, "
    "cost, safety, maintenance, and forecasting, powered by machine learning."
)

try:
    ops_df, ore_df, ore_source = load_data()
except Exception as exc:  # noqa: BLE001 — show a friendly error, never a traceback page
    st.error(
        "Could not load the dataset. Run `python src/data_generator.py` first, "
        f"then reload. Details: {exc}"
    )
    st.stop()

render_sidebar(ops_df)

st.caption(f"Ore quality data source: **{ore_source}**")

col1, col2, col3 = st.columns(3)
col1.metric("Sites", ops_df["site_id"].nunique())
col2.metric("Equipment units", ops_df["equipment_id"].nunique())
col3.metric("Records (equipment-days)", f"{len(ops_df):,}")

st.markdown("### Navigate")
st.markdown(
    """
- **Executive** — KPI overview and alerts
- **Production** — output vs target, efficiency, ore quality
- **Equipment** — utilization, fuel efficiency, fleet status
- **Cost** — cost breakdown and cost per ton
- **Safety** — incidents and safety-related downtime
- **Maintenance** — history, costs, preventive schedule
- **Forecast** — production outlook (baseline; ML models in Phase 2)
"""
)
st.info("👈 Use the sidebar filters and the page navigation above to explore.")
