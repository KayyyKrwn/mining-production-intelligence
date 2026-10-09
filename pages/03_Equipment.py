import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard_data import load_data, apply_filters
from src.ui import render_sidebar, empty_state
from src.features import make_anomaly_frame
from src.models.inference import load_model
st.set_page_config(page_title="Equipment", page_icon="⛏️", layout="wide")
ops_df, ore_df, ore_source = load_data()
filters = render_sidebar(ops_df)
df = apply_filters(ops_df, filters)
if df.empty:
    empty_state(); st.stop()

st.title("Equipment Performance")
st.caption("Utilization, fuel efficiency and fleet status across the selected period.")

# ---- Per-equipment aggregates ---------------------------------------------
equip = df.groupby(["equipment_id", "equipment_type", "site_id"]).agg(
    op_hours=("operating_hours", "sum"),
    down_hours=("downtime_hours", "sum"),
    fuel=("fuel_liter", "sum"),
    production=("production_ton", "sum"),
    age=("age_years", "mean"),
).reset_index()

# Utilization with zero-denominator guard (row-wise)
def _util(row):
    denom = row["op_hours"] + row["down_hours"]
    if denom > 0:
        return row["op_hours"] / denom * 100.0
    return float("nan")

equip["utilization_pct"] = equip.apply(_util, axis=1)

# Fuel efficiency (L/ton) with zero-production guard
def _eff(row):
    if row["production"] > 0:
        return row["fuel"] / row["production"]
    return float("nan")

equip["fuel_l_per_ton"] = equip.apply(_eff, axis=1)

# ---- Utilization ranking ---------------------------------------------------
st.subheader("Utilization Ranking")
util_df = equip.dropna(subset=["utilization_pct"]).copy()
if util_df.empty:
    st.info("No utilization data for the current selection.")
else:
    top_util = util_df.nlargest(20, "utilization_pct").sort_values("utilization_pct", ascending=True)
    fig = px.bar(
        top_util,
        x="utilization_pct",
        y="equipment_id",
        color="equipment_type",
        orientation="h",
        title="Top 20 Units by Utilization (%)",
        labels={
            "utilization_pct": "Utilization (%)",
            "equipment_id": "Equipment",
            "equipment_type": "Type",
        },
        hover_data={"site_id": True, "age": ":.1f"},
    )
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=top_util["equipment_id"].tolist()))
    st.plotly_chart(fig, width="stretch")

# ---- Fleet status donut -----------------------------------------------------
st.subheader("Fleet Status")
total_op = df["operating_hours"].sum()
total_down = df["downtime_hours"].sum()
if (total_op + total_down) <= 0:
    st.info("No operating/downtime hours recorded for the current selection.")
else:
    fleet_df = pd.DataFrame(
        {"status": ["Operating", "Downtime"], "hours": [total_op, total_down]}
    )
    fig = px.pie(
        fleet_df,
        names="status",
        values="hours",
        hole=0.4,
        title="Fleet Hours Split",
        color="status",
        color_discrete_map={"Operating": "#2ecc71", "Downtime": "#e74c3c"},
    )
    st.plotly_chart(fig, width="stretch")

# ---- Fuel efficiency ranking ------------------------------------------------
st.subheader("Fuel Efficiency Ranking")
eff_df = equip.dropna(subset=["fuel_l_per_ton"]).copy()
if eff_df.empty:
    st.info("No fuel efficiency data for the current selection.")
else:
    best_eff = eff_df.nsmallest(15, "fuel_l_per_ton").sort_values("fuel_l_per_ton", ascending=False)
    fig = px.bar(
        best_eff,
        x="fuel_l_per_ton",
        y="equipment_id",
        color="equipment_type",
        orientation="h",
        title="Best 15 Units by Fuel Efficiency (L/ton — lower is better)",
        labels={
            "fuel_l_per_ton": "Fuel (L/ton)",
            "equipment_id": "Equipment",
            "equipment_type": "Type",
        },
        hover_data={"site_id": True, "production": ":.1f"},
    )
    st.plotly_chart(fig, width="stretch")

# ---- Age vs fuel efficiency scatter -----------------------------------------
st.subheader("Age vs Fuel Efficiency")
scatter_df = equip.dropna(subset=["fuel_l_per_ton"]).copy()
if scatter_df.empty:
    st.info("No data for age vs fuel efficiency in the current selection.")
else:
    fig = px.scatter(
        scatter_df,
        x="age",
        y="fuel_l_per_ton",
        color="equipment_type",
        hover_data={"equipment_id": True, "site_id": True},
        title="Equipment Age vs Fuel Efficiency",
        labels={
            "age": "Age (years)",
            "fuel_l_per_ton": "Fuel (L/ton)",
            "equipment_type": "Type",
        },
    )
    st.plotly_chart(fig, width="stretch")

# ---- Recent anomaly flags (ML, Phase 2) ------------------------------------
st.subheader("Recent Anomaly Flags")

try:
    anomaly_payload = load_model("anomaly")
except FileNotFoundError:
    st.info(
        "Anomaly-detection model is not trained yet — this section is hidden. "
        "Train it with: python scripts/train_all.py"
    )
else:
    anomaly_frame = make_anomaly_frame(df)
    if anomaly_frame.empty:
        st.info("No equipment-day data to score for anomalies in the current selection.")
    else:
        iso_model = anomaly_payload["model"]
        features = anomaly_payload["features"]
        scores = iso_model.decision_function(
            anomaly_frame[features].to_numpy(dtype=float)
        )
        anomaly_frame = anomaly_frame.assign(anomaly_score=scores)
        # Model's own threshold: IsolationForest flags decision_function < 0.
        flagged = anomaly_frame[anomaly_frame["anomaly_score"] < 0].copy()

        max_date = anomaly_frame["date"].max()
        recent_flags = flagged[
            flagged["date"] >= max_date - pd.Timedelta(days=13)
        ].sort_values(["date", "anomaly_score"])

        if recent_flags.empty:
            st.info("No anomaly flags for any unit in the last 14 days. ✅")
        else:
            flag_table = (
                recent_flags[
                    ["date", "equipment_id", "equipment_type", "site_id", "anomaly_score"]
                ]
                .rename(columns={
                    "date": "Date",
                    "equipment_id": "Equipment",
                    "equipment_type": "Type",
                    "site_id": "Site",
                    "anomaly_score": "Anomaly score",
                })
            )
            flag_table["Date"] = pd.to_datetime(
                flag_table["Date"]
            ).dt.strftime("%Y-%m-%d")
            flag_table.insert(1, "Anomaly", "🚨")
            st.dataframe(flag_table, width="stretch", hide_index=True)
            st.caption(
                "Units flagged by the IsolationForest model (anomaly score < 0) "
                "in the last 14 days of the selected period. Lower score = more unusual."
            )
