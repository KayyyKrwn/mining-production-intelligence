import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard_data import load_data, apply_filters
from src.ui import render_sidebar, empty_state
st.set_page_config(page_title="Safety", page_icon="⛏️", layout="wide")
ops_df, ore_df, ore_source = load_data()
filters = render_sidebar(ops_df)
df = apply_filters(ops_df, filters)
if df.empty:
    empty_state(); st.stop()

# ---------------------------------------------------------------- Safety KPIs
st.title("Safety")

safety = df[df["downtime_reason"] == "safety"]

incidents = len(safety)
total_operating_hours = df["operating_hours"].sum()
incident_rate = (incidents / total_operating_hours * 100000) if total_operating_hours > 0 else 0.0
safety_downtime_hours = safety["downtime_hours"].sum() if not safety.empty else 0.0

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Safety incidents", f"{incidents:,}")
kpi2.metric("Incident rate (per 100k hours)", f"{incident_rate:,.2f}")
kpi3.metric("Safety downtime hours", f"{safety_downtime_hours:,.1f}")

# ---------------------------------------------------------------- Monthly trend
st.subheader("Monthly safety incidents")

if safety.empty:
    st.info("No safety incidents in the selected period.")
else:
    monthly = (
        safety.groupby(pd.Grouper(key="date", freq="ME"))
        .size()
        .reset_index(name="incidents")
    )
    monthly = monthly[monthly["incidents"] > 0]
    if monthly.empty:
        st.info("No safety incidents in the selected period.")
    else:
        monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
        fig = px.bar(
            monthly,
            x="month",
            y="incidents",
            labels={"month": "Month", "incidents": "Safety incidents"},
            title="Safety incidents per month",
        )
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Downtime by site
st.subheader("Safety downtime by site")

if safety.empty:
    st.info("No safety downtime recorded in the selected period.")
else:
    by_site = (
        safety.groupby("site_id")["downtime_hours"]
        .sum()
        .reset_index()
        .sort_values("downtime_hours", ascending=True)
    )
    by_site = by_site[by_site["downtime_hours"] > 0]
    if by_site.empty:
        st.info("No safety downtime recorded in the selected period.")
    else:
        fig = px.bar(
            by_site,
            x="downtime_hours",
            y="site_id",
            orientation="h",
            labels={"downtime_hours": "Safety downtime hours", "site_id": "Site"},
            title="Safety downtime hours by site",
        )
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Recent events table
st.subheader("Recent safety events")

if safety.empty:
    st.info("No safety events in the selected period.")
else:
    recent = (
        safety.sort_values("date", ascending=False)
        .head(15)
        .loc[:, ["date", "site_id", "equipment_id", "equipment_type", "downtime_hours"]]
    )
    recent = recent.copy()
    recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(recent, width="stretch", hide_index=True)
