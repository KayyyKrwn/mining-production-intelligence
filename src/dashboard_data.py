"""Shared data loading for the dashboard (cached)."""

import streamlit as st
from pathlib import Path

from src.data_loader import load_operations, load_ore_quality

REPO_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data
def load_data():
    """Load operations + ore quality data once per session.

    Returns (ops_df, ore_df, ore_source) where ore_source is
    "Kaggle" or "Synthetic".
    """
    ops = load_operations(str(REPO_ROOT / "data" / "mining_operations.csv"))
    ore, source = load_ore_quality(
        str(REPO_ROOT / "data" / "raw" / "flotation.csv"),
        str(REPO_ROOT / "data" / "ore_quality_synthetic.csv"),
    )
    return ops, ore, source


def apply_filters(ops_df, filters):
    """Apply global sidebar filters to the operations DataFrame."""
    df = ops_df
    mask = (
        (df["date"] >= filters["start_date"])
        & (df["date"] <= filters["end_date"])
        & (df["site_id"].isin(filters["sites"]))
        & (df["equipment_type"].isin(filters["equipment_types"]))
    )
    return df.loc[mask].copy()
