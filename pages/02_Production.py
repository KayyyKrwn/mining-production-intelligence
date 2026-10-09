"""Production page — output vs target, efficiency, and ore quality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error

from src.dashboard_data import load_data, apply_filters
from src.ui import render_sidebar, empty_state
from src.analytics import production_efficiency
from src.features import make_ore_frame
from src.models.inference import load_model

st.set_page_config(page_title="Production", page_icon="⛏️", layout="wide")

try:
    ops_df, ore_df, ore_source = load_data()
except Exception as exc:  # noqa: BLE001 — friendly error, never a traceback page
    st.error(f"Could not load the dataset. Details: {exc}")
    st.stop()

filters = render_sidebar(ops_df)
df = apply_filters(ops_df, filters)
if df.empty:
    empty_state()
    st.stop()

st.title("⛏️ Production")
st.markdown("Daily output per site, actual production vs target, and ore quality.")

# ---------------------------------------------------------------------------
# 1. Production trend per site: daily sum(production_ton) grouped by date/site
# ---------------------------------------------------------------------------
st.subheader("Daily production trend per site")
daily_site = (
    df.groupby(["date", "site_id"], as_index=False)["production_ton"].sum()
)
if not daily_site.empty:
    fig_trend = px.line(
        daily_site,
        x="date",
        y="production_ton",
        color="site_id",
        markers=True,
        labels={
            "date": "Date",
            "production_ton": "Production (tons)",
            "site_id": "Site",
        },
        title="Daily production by site (tons)",
    )
    st.plotly_chart(fig_trend, width="stretch")
else:
    st.info("No daily production data for the selected filters.")

# ---------------------------------------------------------------------------
# 2. Actual vs target per site (bars) + overall efficiency text line
# ---------------------------------------------------------------------------
st.subheader("Actual vs target production")
per_site = (
    df.groupby("site_id", as_index=False)
    .agg(actual=("production_ton", "sum"), target=("target_ton", "sum"))
)
overall_eff = production_efficiency(df)
st.write(
    f"Overall production efficiency: **{overall_eff:,.1f}%** "
    f"(actual {df['production_ton'].sum():,.0f} t / "
    f"target {df['target_ton'].sum():,.0f} t)"
)
if not per_site.empty:
    per_site_melted = per_site.melt(
        id_vars="site_id",
        value_vars=["actual", "target"],
        var_name="measure",
        value_name="tons",
    )
    fig_avt = px.bar(
        per_site_melted,
        x="site_id",
        y="tons",
        color="measure",
        barmode="group",
        labels={
            "site_id": "Site",
            "tons": "Tons",
            "measure": "",
        },
        title="Actual vs target production per site (tons)",
    )
    st.plotly_chart(fig_avt, width="stretch")
else:
    st.info("No actual/target data for the selected filters.")

# ---------------------------------------------------------------------------
# 3. Efficiency by site: horizontal bar, computed per site via analytics fn
# ---------------------------------------------------------------------------
st.subheader("Efficiency by site")
eff_rows = []
for site, site_df in df.groupby("site_id"):
    eff_rows.append(
        {"site_id": site, "efficiency_pct": production_efficiency(site_df)}
    )
eff_by_site = pd.DataFrame(eff_rows).sort_values("efficiency_pct")
if not eff_by_site.empty:
    fig_eff = px.bar(
        eff_by_site,
        x="efficiency_pct",
        y="site_id",
        orientation="h",
        labels={
            "efficiency_pct": "Production efficiency (%)",
            "site_id": "Site",
        },
        title="Production efficiency per site (%)",
        text_auto=".1f",
    )
    st.plotly_chart(fig_eff, width="stretch")
else:
    st.info("No efficiency data for the selected filters.")

# ---------------------------------------------------------------------------
# 4. Daily production heatmap (date x site); weekly resample if > 120 days
# ---------------------------------------------------------------------------
st.subheader("Daily production heatmap")
heatmap_df = daily_site.copy()
if not heatmap_df.empty:
    date_min = heatmap_df["date"].min()
    date_max = heatmap_df["date"].max()
    span_days = (
        (pd.Timestamp(date_max) - pd.Timestamp(date_min)).days
        if pd.notna(date_min) and pd.notna(date_max)
        else 0
    )
    if span_days > 120:
        heatmap_df = heatmap_df.copy()
        heatmap_df["week"] = pd.to_datetime(heatmap_df["date"]).dt.to_period("W")
        heatmap_df = (
            heatmap_df.groupby(["week", "site_id"], as_index=False)["production_ton"]
            .sum()
            .rename(columns={"week": "period"})
        )
        index_label = "Week"
    else:
        heatmap_df = heatmap_df.rename(columns={"date": "period"})
        index_label = "Date"
    pivot = heatmap_df.pivot_table(
        index="period", columns="site_id", values="production_ton", aggfunc="sum"
    )
    if not pivot.empty:
        st.caption(f"Rows: {index_label}; cells: tons produced.")
        fig_hm = px.imshow(
            pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            labels=dict(x="Site", y=index_label, color="Tons"),
            color_continuous_scale="YlGn",
            aspect="auto",
        )
        fig_hm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_hm, width="stretch")
    else:
        st.info("No data to build the heatmap for the selected filters.")
else:
    st.info("No daily production data for the selected filters.")

# ---------------------------------------------------------------------------
# 5. Ore quality panel: daily mean silica_pct per (date, site_id)
# ---------------------------------------------------------------------------
st.subheader("Ore quality")
ore = ore_df.copy()
if "silica_pct" in ore.columns and not ore.empty:
    ore["date"] = pd.to_datetime(ore["date"]).dt.floor("D")
    if "site_id" not in ore.columns:
        ore["site_id"] = "Flotation Plant"
    ore_daily = (
        ore.groupby(["date", "site_id"], as_index=False)["silica_pct"].mean()
    )
    if not ore_daily.empty:
        fig_silica = px.line(
            ore_daily,
            x="date",
            y="silica_pct",
            color="site_id",
            markers=True,
            labels={
                "date": "Date",
                "silica_pct": "Silica (%)",
                "site_id": "Site",
            },
            title="Daily mean silica percentage in concentrate",
        )
        st.plotly_chart(fig_silica, width="stretch")
    else:
        st.info("No ore quality data available.")
else:
    st.info("No ore quality data available.")
st.caption(f"Data source: {ore_source}")

# ---------------------------------------------------------------------------
# 6. Ore quality model predictions (holdout): predicted vs actual silica_pct
# ---------------------------------------------------------------------------
st.subheader("Ore quality — model predictions")

try:
    ore_bundle = load_model("ore_quality")
except FileNotFoundError:
    st.info(
        "The ore quality prediction model has not been trained yet. "
        "Run `python scripts/train_all.py` from the repository root to train "
        "it and see model predictions here."
    )
    st.caption("Showing the Phase 1 view above (observed ore quality only).")
else:
    if isinstance(ore_bundle, dict):
        ore_model = ore_bundle["model"]
        ore_feats = list(ore_bundle.get("feature_columns", []))
        ore_target = ore_bundle.get("target", "silica_pct")
        trained_on = ore_bundle.get("source", "other")
    else:  # pragma: no cover — the train script always saves a bundle dict
        ore_model, ore_feats, ore_target, trained_on = ore_bundle, [], "silica_pct", "other"

    ore_frame = (
        make_ore_frame(ore_df)
        .dropna(subset=[ore_target])
        .sort_values("date")
        .reset_index(drop=True)
    )
    missing_feats = [c for c in ore_feats if c not in ore_frame.columns]
    if missing_feats:
        st.info(
            "The trained ore quality model expects sensor columns that are "
            f"not in the current ore dataset (model trained on {trained_on} "
            f"data, dashboard is using the {ore_source} dataset). "
            "Showing the Phase 1 view above."
        )
    elif not ore_feats:
        st.info("The trained model does not record feature columns.")
    else:
        cut = int(len(ore_frame) * 0.8)
        holdout = ore_frame.iloc[cut:]
        if holdout.empty:
            st.info("Not enough ore data to evaluate the model on a holdout set.")
        else:
            actual = holdout[ore_target].to_numpy()
            predicted = ore_model.predict(holdout[ore_feats])
            rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
            baseline = float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        np.full(len(actual), ore_frame[ore_target].iloc[:cut].mean()),
                    )
                )
            )
            cmp_df = pd.DataFrame(
                {
                    "date": holdout["date"].to_numpy(),
                    "Actual": actual,
                    "Predicted": predicted,
                }
            )
            fig_pred = px.line(
                cmp_df,
                x="date",
                y=["Actual", "Predicted"],
                labels={
                    "date": "Date",
                    "value": "Silica (%)",
                    "variable": "",
                },
                title="Predicted vs actual silica percentage "
                "(holdout: last 20% of hours)",
            )
            st.plotly_chart(fig_pred, width="stretch")
            col_a, col_b = st.columns(2)
            col_a.metric("Model RMSE (holdout)", f"{rmse:.3f} %")
            col_b.metric("Baseline RMSE (train mean)", f"{baseline:.3f} %")
st.badge(f"Ore data source: {ore_source}")
