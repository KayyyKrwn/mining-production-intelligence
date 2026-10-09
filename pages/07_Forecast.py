import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard_data import load_data, apply_filters
from src.models.inference import load_model
from src.ui import render_sidebar, empty_state
st.set_page_config(page_title="Forecast", page_icon="⛏️", layout="wide")
ops_df, ore_df, ore_source = load_data()
filters = render_sidebar(ops_df)
df = apply_filters(ops_df, filters)
if df.empty:
    empty_state(); st.stop()

st.title("Forecast")

horizon_label = st.radio("Forecast horizon", ["30 days", "90 days"], horizontal=True)
horizon_days = int(horizon_label.split()[0])

# Daily production per site from the filtered data.
daily = (
    df.groupby(["date", "site_id"], as_index=False)["production_ton"].sum()
)
daily["date"] = pd.to_datetime(daily["date"])
max_date = daily["date"].max()
n_history_days = int(daily["date"].nunique())
if n_history_days < 14:
    empty_state("Not enough history for a baseline forecast.")
    st.stop()

future_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=horizon_days)


# ---------------------------------------------------------------------------
# Phase 1 baseline (kept verbatim): 30-day mean x seasonal month factor.
# ---------------------------------------------------------------------------
def _phase1_forecast(daily, max_date, future_dates):
    """Return the Phase 1 baseline forecast frame."""
    window_start = max_date - pd.Timedelta(days=29)
    recent = daily[daily["date"] >= window_start]
    baseline = recent.groupby("site_id")["production_ton"].mean()
    baseline = baseline.where(baseline.notna() & (baseline != float("inf")), 0.0).fillna(0.0)

    forecast_months = sorted(set(future_dates.month))
    hist = daily.copy()
    hist["month"] = hist["date"].dt.month
    month_mean = hist[hist["month"].isin(forecast_months)].groupby("site_id")["production_ton"].mean()
    overall_mean = hist.groupby("site_id")["production_ton"].mean()

    seasonal = pd.Series(1.0, index=baseline.index, dtype=float)
    for site in seasonal.index:
        m = month_mean.get(site, float("nan"))
        o = overall_mean.get(site, float("nan"))
        if pd.notna(m) and pd.notna(o) and o > 0:
            factor = m / o
            seasonal[site] = factor if pd.notna(factor) and factor != float("inf") else 1.0

    forecast_rows = []
    for site in baseline.index:
        daily_value = baseline[site] * seasonal[site]
        for d in future_dates:
            forecast_rows.append({"date": d, "site_id": site, "production_ton": daily_value})
    return pd.DataFrame(forecast_rows)


def _render_phase1_section(daily, max_date, future_dates, horizon_days):
    """Render the original Phase 1 baseline chart + summary table."""
    st.warning("Simple baseline projection — ML production-forecasting model arrives in Phase 2.")
    forecast = _phase1_forecast(daily, max_date, future_dates)

    hist_180 = daily[daily["date"] >= max_date - pd.Timedelta(days=179)].copy()
    plot_df = None
    if not hist_180.empty and not forecast.empty:
        hist_180["kind"] = "History"
        fc = forecast.copy()
        fc["kind"] = "Forecast"
        plot_df = pd.concat([hist_180, fc], ignore_index=True)

    if plot_df is not None and not plot_df.empty:
        fig = px.line(
            plot_df,
            x="date",
            y="production_ton",
            color="site_id",
            line_dash="kind",
            labels={
                "date": "Date",
                "production_ton": "Daily production (ton)",
                "site_id": "Site",
                "kind": "Series",
            },
            title="Daily production: last 180 days vs baseline forecast",
        )
        for trace in fig.data:
            if "Forecast" in str(trace.name):
                trace.line.update(dash="dash", width=3.5)
            else:
                trace.line.update(width=1.25)
        fig.add_vline(
            x=future_dates[0].to_pydatetime(),
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast start",
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "History = observed daily production per site (thin lines). "
            "Forecast = per-site baseline (30-day mean) × seasonal month factor (dashed lines)."
        )
    else:
        empty_state("No data available to draw the forecast chart.")

    if not forecast.empty:
        _render_summary_table(
            forecast, f"Forecast summary — next {horizon_days} days", horizon_days
        )


def _render_summary_table(forecast, title, horizon_days, forecast_ml=None):
    """Per-site forecast summary; optionally alongside the ML model forecast."""
    def _totals(frame, col_name):
        per_site = (
            frame.groupby("site_id")["production_ton"]
            .sum()
            .reset_index()
            .rename(columns={"site_id": "Site", "production_ton": col_name})
        )
        per_site[f"{col_name} — avg/day"] = per_site[col_name] / horizon_days
        return per_site

    base = _totals(forecast, "Baseline total (ton)")
    if forecast_ml is not None and not forecast_ml.empty:
        ml = _totals(forecast_ml, "ML model total (ton)")
        table = base.merge(ml, on="Site", how="outer")
        table = table[
            ["Site", "ML model total (ton)", "ML model total (ton) — avg/day",
             "Baseline total (ton)", "Baseline total (ton) — avg/day"]
        ]
        total_ml = forecast_ml["production_ton"].sum()
        total_base = forecast["production_ton"].sum()
        total_row = pd.DataFrame([{
            "Site": "ALL SITES",
            "ML model total (ton)": total_ml,
            "ML model total (ton) — avg/day": total_ml / horizon_days,
            "Baseline total (ton)": total_base,
            "Baseline total (ton) — avg/day": total_base / horizon_days,
        }])
        fmt = {
            "ML model total (ton)": "{:,.0f}",
            "ML model total (ton) — avg/day": "{:,.1f}",
            "Baseline total (ton)": "{:,.0f}",
            "Baseline total (ton) — avg/day": "{:,.1f}",
        }
    else:
        table = base.rename(columns={
            "Baseline total (ton)": "Total forecasted (ton)",
            "Baseline total (ton) — avg/day": "Avg per day (ton)",
        })
        total_ton = forecast["production_ton"].sum()
        total_row = pd.DataFrame([{
            "Site": "ALL SITES",
            "Total forecasted (ton)": total_ton,
            "Avg per day (ton)": total_ton / horizon_days,
        }])
        fmt = {"Total forecasted (ton)": "{:,.0f}", "Avg per day (ton)": "{:,.1f}"}

    table = pd.concat([table, total_row], ignore_index=True)
    st.subheader(title)
    st.dataframe(table.style.format(fmt), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Phase 2 ML forecast: recursive multi-step prediction per site.
# ---------------------------------------------------------------------------
def _recursive_forecast(bundle, df, future_dates):
    """Roll the trained model forward one day at a time per site.

    Lag/rolling features are recomputed from observed + already-predicted
    values; calendar features from the future date; exogenous inputs from the
    site's last-30-day means (same scale as training).
    """
    frame = (
        df.groupby(["site_id", "date"], as_index=False)
        .agg(
            production_ton=("production_ton", "sum"),
            temp_c=("temp_c", "mean"),
            rainfall_mm=("rainfall_mm", "sum"),
            n_units=("equipment_id", "nunique"),
        )
    )
    frame["date"] = pd.to_datetime(frame["date"])
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    rows = []
    for site, g in frame.groupby("site_id"):
        g = g.sort_values("date")
        if len(g) < 31:
            continue  # not enough history for lag_30
        hist = list(g["production_ton"].to_numpy(dtype=float))
        tail = g.tail(30)
        f_temp = float(tail["temp_c"].mean())
        f_rain = float(tail["rainfall_mm"].mean())
        f_units = float(tail["n_units"].mean())
        for d in future_dates:
            feats = {
                "lag_1": hist[-1],
                "lag_7": hist[-7],
                "lag_14": hist[-14],
                "lag_30": hist[-30],
                "roll_mean_7": float(np.mean(hist[-7:])),
                "roll_mean_30": float(np.mean(hist[-30:])),
                "dow": d.dayofweek,
                "month": d.month,
                "temp_c": f_temp,
                "rainfall_mm": f_rain,
                "n_units": f_units,
            }
            X = pd.DataFrame([feats])[feature_cols]
            yhat = max(float(model.predict(X)[0]), 0.0)
            hist.append(yhat)
            rows.append({"date": d, "site_id": site, "production_ton": yhat})
    return pd.DataFrame(rows)


# --- Load the trained model (guard: fall back to the Phase 1 view) ---
try:
    bundle = load_model("forecast")
except FileNotFoundError:
    st.info("Model not trained yet — run: python scripts/train_all.py")
    _render_phase1_section(daily, max_date, future_dates, horizon_days)
    st.stop()

metrics = bundle["metrics"]

# --- Holdout performance: model vs naive baseline ---
st.subheader("Model performance (chronological holdout — last 20% of history)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Holdout MAE — ML model", f"{metrics['mae']:,.0f} ton")
col2.metric("Holdout MAE — naive (7d ago)", f"{metrics['baseline_mae']:,.0f} ton")
col3.metric("Holdout RMSE — ML model", f"{metrics['rmse']:,.0f} ton")
col4.metric("Holdout RMSE — naive (7d ago)", f"{metrics['baseline_rmse']:,.0f} ton")
improvement = (
    (metrics["baseline_mae"] - metrics["mae"]) / metrics["baseline_mae"] * 100
    if metrics["baseline_mae"] > 0
    else 0.0
)
st.caption(
    f"Gradient-boosting regressor beats the naive baseline by {improvement:.1f}% "
    "on holdout MAE (naive baseline = repeat the value from 7 days ago)."
)

# --- ML forecast + Phase 1 baseline, side by side ---
ml_forecast = _recursive_forecast(bundle, df, future_dates)
phase1_forecast = _phase1_forecast(daily, max_date, future_dates)

hist_180 = daily[daily["date"] >= max_date - pd.Timedelta(days=179)].copy()
plot_df = None
if not hist_180.empty and not ml_forecast.empty:
    hist_180["kind"] = "History"
    fc_ml = ml_forecast.copy()
    fc_ml["kind"] = "ML forecast"
    fc_b = phase1_forecast.copy()
    fc_b["kind"] = "Baseline (Phase 1)"
    plot_df = pd.concat([hist_180, fc_ml, fc_b], ignore_index=True)

if plot_df is not None and not plot_df.empty:
    fig = px.line(
        plot_df,
        x="date",
        y="production_ton",
        color="site_id",
        line_dash="kind",
        labels={
            "date": "Date",
            "production_ton": "Daily production (ton)",
            "site_id": "Site",
            "kind": "Series",
        },
        title="Daily production: last 180 days vs ML forecast vs Phase 1 baseline",
    )
    for trace in fig.data:
        name = str(trace.name)
        if "ML forecast" in name:
            trace.line.update(width=3.0)  # solid, bold
        elif "Baseline" in name:
            trace.line.update(dash="dash", width=2.0)
        else:
            trace.line.update(width=1.25)  # thin history
    fig.add_vline(
        x=future_dates[0].to_pydatetime(),
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast start",
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "History = observed daily production per site (thin lines). "
        "ML forecast = gradient-boosting model rolled forward recursively (bold solid lines). "
        "Baseline = Phase 1 projection — 30-day mean × seasonal month factor (dashed lines)."
    )
else:
    empty_state("No data available to draw the forecast chart.")

# --- Summary table: ML vs baseline ---
if not phase1_forecast.empty:
    _render_summary_table(
        phase1_forecast,
        f"Forecast summary — next {horizon_days} days (ML model vs Phase 1 baseline)",
        horizon_days,
        forecast_ml=ml_forecast,
    )
