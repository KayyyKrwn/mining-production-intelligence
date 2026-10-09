"""Shared sidebar filters and UI helpers for all dashboard pages."""

import pandas as pd
import streamlit as st


def render_sidebar(ops_df: pd.DataFrame) -> dict:
    """Render global filters in the sidebar. Returns a filters dict with:
    start_date, end_date (pd.Timestamp), sites (list), equipment_types (list).
    """
    st.sidebar.header("Filters")

    min_date = pd.Timestamp(ops_df["date"].min()).date()
    max_date = pd.Timestamp(ops_df["date"].max()).date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    else:  # single date selected -> treat as one day
        start_date = end_date = pd.Timestamp(date_range)

    all_sites = sorted(ops_df["site_id"].unique().tolist())
    sites = st.sidebar.multiselect("Site", options=all_sites, default=all_sites)

    all_types = sorted(ops_df["equipment_type"].unique().tolist())
    equipment_types = st.sidebar.multiselect(
        "Equipment type", options=all_types, default=all_types
    )

    filters = {
        "start_date": start_date,
        "end_date": end_date,
        "sites": sites,
        "equipment_types": equipment_types,
    }
    st.session_state["filters"] = filters
    return filters


def empty_state(message: str = "No data for the selected filters. Try widening the date range or selecting more sites/equipment."):
    """Friendly empty state shown when a filter yields zero rows."""
    st.info(message)
