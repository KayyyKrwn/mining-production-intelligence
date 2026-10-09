import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard_data import load_data, apply_filters
from src.ui import render_sidebar, empty_state
from src.analytics import cost_per_ton
from src.features import make_cost_frame
from src.models.inference import load_model
from src.models.train_cost import build_cost_matrix, time_split
st.set_page_config(page_title="Cost", page_icon="⛏️", layout="wide")
ops_df, ore_df, ore_source = load_data()
filters = render_sidebar(ops_df)
df = apply_filters(ops_df, filters)
if df.empty:
    empty_state(); st.stop()

# ---------------------------------------------------------------- Title + KPI
st.title("⛏️ Cost")
st.caption("Operating cost breakdown, composition, and cost-efficiency trends for the selected period.")

overall_cost_per_ton = cost_per_ton(df)
st.metric("Overall cost per ton", f"${overall_cost_per_ton:,.2f}/ton")

# --------------------------------------- ML: operating cost prediction
st.subheader("Operating cost prediction (machine learning)")
try:
    _bundle = load_model("cost")
except FileNotFoundError:
    st.info(
        "The cost prediction model has not been trained yet, so this section "
        "shows the Phase 1 analytics instead. Train the model with "
        "`python scripts/train_all.py` to unlock the prediction chart."
    )
else:
    _model = _bundle["model"] if isinstance(_bundle, dict) else _bundle
    _cols = _bundle.get("feature_columns") if isinstance(_bundle, dict) else None
    _frame = make_cost_frame(ops_df)
    _X, _y = build_cost_matrix(_frame)
    if _cols is not None:
        _X = _X.reindex(columns=_cols, fill_value=0.0)
    _X_train, _X_test, _y_train, _y_test = time_split(_X, _y)
    _pred = _model.predict(_X_test)
    _dates = pd.to_datetime(_frame["date"].iloc[_X_test.index])
    _holdout = pd.DataFrame({
        "Date": _dates.values,
        "Actual cost": _y_test.values,
        "Predicted cost": _pred,
    }).sort_values("Date")
    _mae = float(abs(_holdout["Actual cost"] - _holdout["Predicted cost"]).mean())
    _baseline_mae = float(abs(_holdout["Actual cost"] - _y_train.mean()).mean())
    col_mae, col_r2 = st.columns(2)
    col_mae.metric(
        "Holdout MAE",
        f"${_mae:,.0f}",
        delta=f"{100.0 * (_baseline_mae - _mae) / _baseline_mae:.1f}% vs naive baseline",
        delta_color="inverse",
    )
    _stored = _bundle.get("metrics") if isinstance(_bundle, dict) else None
    col_r2.metric(
        "Holdout R²",
        f"{_stored['r2']:.4f}" if _stored else "n/a",
        help="R² on the last 20% of the time-ordered holdout set.",
    )
    fig_pred = px.line(
        _holdout.melt(id_vars="Date", var_name="Series", value_name="USD"),
        x="Date", y="USD", color="Series",
        title="Predicted vs actual operating cost (holdout, site-day)",
        labels={"Date": "Date", "USD": "USD"},
    )
    st.plotly_chart(fig_pred, width="stretch")
    st.caption(
        "Gradient-boosted regression trained on the first 80% of the period "
        "(time-ordered split). Baseline: constant mean of the training target "
        f"(MAE ${_baseline_mae:,.0f})."
    )

# ------------------------------------------------- Cost composition per site
st.subheader("Cost composition by site")
df = df.copy()
df["fuel_cost"] = df["fuel_liter"] * 0.9
df["labor_cost"] = df["workers_on_shift"] * 25.0
df["maintenance_cost"] = df["maintenance_cost_usd"]
df["overhead"] = 500.0

components = ["fuel_cost", "labor_cost", "maintenance_cost", "overhead"]
site_comp = df.groupby("site_id", as_index=False)[components].sum()
if not site_comp.empty:
    comp_long = site_comp.melt(
        id_vars="site_id", value_vars=components,
        var_name="component", value_name="usd",
    )
    comp_long["component"] = comp_long["component"].replace({
        "fuel_cost": "Fuel ($0.90/L)",
        "labor_cost": "Labor ($25/worker-shift)",
        "maintenance_cost": "Maintenance",
        "overhead": "Overhead ($500/day)",
    })
    fig_comp = px.bar(
        comp_long, x="site_id", y="usd", color="component",
        title="Cost composition by site (USD)",
        labels={"site_id": "Site", "usd": "USD", "component": "Component"},
    )
    st.plotly_chart(fig_comp, width="stretch")
else:
    st.info("No cost data available for the selected filters.")
st.caption(
    "Components are derived per record: Fuel = fuel_liter × $0.90, "
    "Labor = workers_on_shift × $25, Maintenance = maintenance_cost_usd, "
    "Overhead = flat $500 per day. Figures above are summed across the selected period."
)

# ------------------------------------------------------ Monthly cost-per-ton
st.subheader("Cost per ton trend")
df["date"] = pd.to_datetime(df["date"])
monthly = (
    df.groupby(pd.Grouper(key="date", freq="ME"))
    .agg(cost=("operating_cost_usd", "sum"), production=("production_ton", "sum"))
    .reset_index()
)
monthly = monthly[(monthly["cost"] != 0) | (monthly["production"] != 0)]
monthly["cost_per_ton"] = monthly.apply(
    lambda r: r["cost"] / r["production"] if r["production"] else 0.0, axis=1
)
if not monthly.empty:
    fig_trend = px.line(
        monthly, x="date", y="cost_per_ton",
        title="Monthly cost per ton (USD/ton)",
        labels={"date": "Month", "cost_per_ton": "USD/ton"},
        markers=True,
    )
    st.plotly_chart(fig_trend, width="stretch")
else:
    st.info("No monthly cost data available for the selected filters.")

# ---------------------------------------------- Top 10 equipment by cost
st.subheader("Top 10 equipment by operating cost")
equip = (
    df.groupby(["equipment_id", "equipment_type", "site_id"], as_index=False)
    .agg(
        total_operating_cost_usd=("operating_cost_usd", "sum"),
        total_production_ton=("production_ton", "sum"),
    )
)
if not equip.empty:
    equip["cost_per_ton"] = equip.apply(
        lambda r: r["total_operating_cost_usd"] / r["total_production_ton"]
        if r["total_production_ton"] else 0.0,
        axis=1,
    )
    top10 = equip.sort_values("total_operating_cost_usd", ascending=False).head(10)
    top10 = top10.rename(columns={
        "equipment_id": "Equipment ID",
        "equipment_type": "Equipment type",
        "site_id": "Site",
        "total_operating_cost_usd": "Total operating cost (USD)",
        "total_production_ton": "Total production (ton)",
        "cost_per_ton": "Cost per ton (USD/ton)",
    })
    st.dataframe(
        top10.style.format({
            "Total operating cost (USD)": "${:,.2f}",
            "Total production (ton)": "{:,.1f}",
            "Cost per ton (USD/ton)": "${:,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No equipment cost data available for the selected filters.")
