import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard_data import load_data, apply_filters
from src.ui import render_sidebar, empty_state
from src.analytics import (production_efficiency, cost_per_ton, equipment_utilization,
                           fuel_efficiency, downtime_summary, productivity)
from src.features import make_anomaly_frame
from src.models.inference import load_model
st.set_page_config(page_title="Executive Dashboard", page_icon="⛏️", layout="wide")
ops_df, ore_df, ore_source = load_data()
filters = render_sidebar(ops_df)
df = apply_filters(ops_df, filters)
if df.empty:
    empty_state(); st.stop()

st.title("⛏️ Executive Dashboard")
st.write("Site-level KPIs, production vs target, and equipment risk signals for mining operations.")

# ---------------------------------------------------------------- KPIs
st.subheader("Key Performance Indicators")

prod_eff = production_efficiency(df)
cpt = cost_per_ton(df)
util = equipment_utilization(df)
fuel_eff = fuel_efficiency(df)
downtime_pct, _breakdown = downtime_summary(df)
prodv = productivity(df)

row1 = st.columns(3)
row1[0].metric("Production Efficiency", f"{prod_eff:.1f} %")
row1[1].metric("Cost per Ton", f"${cpt:,.2f}")
row1[2].metric("Equipment Utilization", f"{util:.1f} %")

row2 = st.columns(3)
row2[0].metric("Fuel Efficiency", f"{fuel_eff:.2f} L/ton")
row2[1].metric("Downtime", f"{downtime_pct:.1f} %")
row2[2].metric("Productivity", f"{prodv:.1f} ton/worker-day")

# ------------------------------------------------- daily production chart
st.subheader("Daily Production vs Target")

daily = (
    df.groupby("date")
    .agg(production=("production_ton", "sum"), target=("target_ton", "sum"))
    .reset_index()
    .sort_values("date")
)

if daily.empty or (daily["production"].eq(0).all() and daily["target"].eq(0).all()):
    st.info("No production data for the selected filters.")
else:
    long = daily.melt(
        id_vars="date", value_vars=["production", "target"],
        var_name="metric", value_name="tons",
    )
    fig = px.line(
        long, x="date", y="tons", color="metric",
        labels={"date": "Date", "tons": "Tons", "metric": "Metric"},
        title="Daily production vs site daily target (sum across equipment)",
    )
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------- top 5 at-risk table
st.subheader("Top 5 Equipment at Risk")
st.warning("Heuristic proxy — ML failure-prediction model arrives in Phase 2.")


def _window_agg(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["equipment_type", "site_id", "fuel", "prod", "mech"]
        )
    return frame.groupby("equipment_id").agg(
        equipment_type=("equipment_type", "first"),
        site_id=("site_id", "first"),
        fuel=("fuel_liter", "sum"),
        prod=("production_ton", "sum"),
        mech=("downtime_reason", lambda s: (s == "mechanical_minor").sum()),
    )


max_date = df["date"].max()
last_start = max_date - pd.Timedelta(days=13)
prior_start = max_date - pd.Timedelta(days=27)

last_w = df[(df["date"] >= last_start) & (df["date"] <= max_date)]
prior_w = df[(df["date"] >= prior_start) & (df["date"] < last_start)]

last_agg = _window_agg(last_w)
prior_agg = _window_agg(prior_w)

if last_agg.empty:
    st.info("Not enough data in the last 14 days to compute risk scores.")
else:
    merged = last_agg.join(prior_agg, how="left", rsuffix="_prior").fillna(0)

    def _fuel_per_ton(row, prefix=""):
        fuel = row[f"fuel{prefix}"]
        prod = row[f"prod{prefix}"]
        return fuel / prod if prod else 0.0

    def _delta(last_val: float, prior_val: float) -> float:
        if prior_val:
            return (last_val - prior_val) / prior_val
        return last_val if last_val else 0.0

    risk = merged.reset_index()
    risk["fuel_last"] = risk.apply(lambda r: _fuel_per_ton(r), axis=1)
    risk["fuel_prior"] = risk.apply(lambda r: _fuel_per_ton(r, "_prior"), axis=1)
    risk["delta_fuel"] = risk.apply(
        lambda r: _delta(r["fuel_last"], r["fuel_prior"]), axis=1
    )
    risk["delta_mech"] = risk["mech"] - risk["mech_prior"]


    def _minmax(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        return (series - lo) / (hi - lo) if hi != lo else pd.Series(0.0, index=series.index)

    risk["norm_fuel"] = _minmax(risk["delta_fuel"])
    risk["norm_mech"] = _minmax(risk["delta_mech"])
    risk["risk_score"] = (risk["norm_fuel"] + risk["norm_mech"]).round(3)

    top5 = (
        risk.sort_values("risk_score", ascending=False)
        .head(5)[["equipment_id", "equipment_type", "site_id", "risk_score"]]
        .rename(columns={
            "equipment_id": "Equipment",
            "equipment_type": "Type",
            "site_id": "Site",
            "risk_score": "Risk score",
        })
    )

    st.dataframe(top5, width="stretch", hide_index=True)
    st.caption(
        "Risk score = min-max normalized sum of (1) fuel liters/ton change and "
        "(2) mechanical_minor downtime count change, comparing the last 14 days "
        "vs the prior 14 days. Higher = larger deterioration."
    )

# ----------------------------------------- recent anomalies (ML, Phase 2)
st.subheader("Recent Anomalies (ML)")

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

        max_date = anomaly_frame["date"].max()
        recent = anomaly_frame[
            anomaly_frame["date"] >= max_date - pd.Timedelta(days=29)
        ]
        if recent.empty:
            st.info("No data in the last 30 days to score for anomalies.")
        else:
            top_anomalies = (
                recent.nsmallest(5, "anomaly_score")[
                    ["date", "equipment_id", "equipment_type", "site_id",
                     "anomaly_score", "fuel_per_ton", "cost_per_ton", "downtime_hours"]
                ]
                .rename(columns={
                    "date": "Date",
                    "equipment_id": "Equipment",
                    "equipment_type": "Type",
                    "site_id": "Site",
                    "anomaly_score": "Anomaly score",
                    "fuel_per_ton": "Fuel (L/ton)",
                    "cost_per_ton": "Cost ($/ton)",
                    "downtime_hours": "Downtime (h)",
                })
            )
            top_anomalies["Date"] = pd.to_datetime(
                top_anomalies["Date"]
            ).dt.strftime("%Y-%m-%d")
            st.dataframe(top_anomalies, width="stretch", hide_index=True)
            st.caption(
                "Top-5 most anomalous equipment-days in the last 30 days from the "
                "IsolationForest model (lower anomaly score = more unusual). "
                f"In-sample precision@k: {anomaly_payload.get('precision_at_k', float('nan')):.2f}."
            )
