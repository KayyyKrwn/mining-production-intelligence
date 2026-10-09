"""Maintenance page: maintenance history, rule-based preventive schedule,
and ML-based 7-day equipment failure-risk prediction (Phase 2).

The ML section loads ``models/failure.pkl`` via ``src.models.inference``.
If the model file is missing, the page falls back to the Phase 1
rule-based view instead of crashing. SHAP explanations are optional:
``shap`` is imported defensively and the explanation is skipped with a
small note when the library is unavailable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import shap
except ImportError:  # optional dependency — explanations are skipped then
    shap = None

from src.dashboard_data import apply_filters, load_data
from src.features import make_failure_frame
from src.models.inference import load_model
from src.ui import empty_state, render_sidebar

st.set_page_config(page_title="Maintenance", page_icon="⛏️", layout="wide")

ops_df, ore_df, ore_source = load_data()
filters = render_sidebar(ops_df)
df = apply_filters(ops_df, filters)
if df.empty:
    empty_state()
    st.stop()

SERVICE_INTERVAL_HOURS = 500

FAILURE_MODEL_NAME = "failure"
# Mirror of src.models.train_failure.FEATURE_COLS; used when the model
# bundle does not carry its own column list.
FAILURE_FEATURE_COLS = [
    "fuel_per_ton",
    "fuel_trend_14d",
    "minor_downtime_14d",
    "hours_since_maintenance",
    "age_years",
    "operating_hours",
]


def _failure_bundle():
    """Load the failure-prediction bundle, or None when not trained yet."""
    try:
        return load_model(FAILURE_MODEL_NAME)
    except FileNotFoundError:
        return None


def _model_and_features(bundle):
    """Unpack a model bundle saved by train_failure.py.

    Accepts the dict bundle (``{"model": ..., "feature_cols": [...]}``) and
    tolerates a bare fitted estimator for forward compatibility.
    """
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("feature_cols", FAILURE_FEATURE_COLS)
    return bundle, FAILURE_FEATURE_COLS


def explain_failure_risk(model, feature_row, feature_names, background, shap_lib=shap):
    """SHAP contributions of one equipment-day row.

    Args:
        model: fitted classifier with ``predict_proba``.
        feature_row: Series of feature values for one equipment-day.
        feature_names: ordered feature names.
        background: background sample DataFrame for the explainer.
        shap_lib: the ``shap`` module, or None to skip explanation.

    Returns:
        ``(names, values)`` tuple, or None when ``shap_lib`` is None or the
        explanation fails for any reason (the page then shows a small note
        instead of crashing).
    """
    if shap_lib is None:
        return None
    try:
        # astype(float): a Series picked via Series[list] can be object-dtype
        # (mixed int64/float64 row), which shap's numba masker rejects.
        proba_fn = lambda X: model.predict_proba(X)[:, 1]  # noqa: E731
        explainer = shap_lib.Explainer(proba_fn, background.astype(float))
        shap_values = explainer(feature_row.astype(float).to_frame().T)
    except Exception:
        return None
    values = np.asarray(shap_values.values).ravel().tolist()
    return list(feature_names), values

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("⛏️ Maintenance")
st.caption(
    "Maintenance history, rule-based preventive schedule, and an ML model "
    "predicting the probability of equipment failure in the next 7 days."
)

# ---------------------------------------------------------------------------
# Section 1: Maintenance history table
# ---------------------------------------------------------------------------
st.subheader("Maintenance history")

history = df[df["maintenance_type"].ne("none")].copy()

units = ["All"] + sorted(history["equipment_id"].unique().tolist())
selected_unit = st.selectbox("Equipment", options=units, key="maint_unit")

shown = history if selected_unit == "All" else history[history["equipment_id"] == selected_unit]
shown = shown.sort_values("date", ascending=False).head(50)

history_cols = [
    "date",
    "site_id",
    "equipment_id",
    "maintenance_type",
    "maintenance_cost_usd",
    "downtime_hours",
    "downtime_reason",
]

if shown.empty:
    empty_state("No maintenance events for the current selection.")
else:
    st.dataframe(
        shown[history_cols],
        width="stretch",
        hide_index=True,
    )
    st.caption(f"Showing {len(shown)} event(s), latest first.")

# ---------------------------------------------------------------------------
# Section 2: Cost by maintenance type (bar chart)
# ---------------------------------------------------------------------------
st.subheader("Maintenance cost by type")

cost_by_type = (
    history.groupby("maintenance_type")["maintenance_cost_usd"]
    .sum()
    .reset_index()
    .sort_values("maintenance_cost_usd", ascending=False)
)

if cost_by_type.empty:
    empty_state("No maintenance cost recorded for the current filters.")
else:
    fig_cost = px.bar(
        cost_by_type,
        x="maintenance_type",
        y="maintenance_cost_usd",
        title="Total maintenance cost by type (USD)",
        labels={"maintenance_type": "Maintenance type", "maintenance_cost_usd": "Cost (USD)"},
    )
    st.plotly_chart(fig_cost, width="stretch")
    st.caption("Rows with maintenance_type = 'none' are excluded (they have zero cost).")

# ---------------------------------------------------------------------------
# Section 3: Preventive schedule — equipment due for service
# ---------------------------------------------------------------------------
st.subheader("Preventive schedule")

df_sorted = df.sort_values("date")
rows = []
for equipment_id, group in df_sorted.groupby("equipment_id"):
    group = group.sort_values("date")
    serviced = group[group["maintenance_type"].isin(["preventive", "corrective"])]
    if serviced.empty:
        hours_since = group["operating_hours"].sum()
    else:
        last_service_date = serviced["date"].max()
        hours_since = group[group["date"] > last_service_date]["operating_hours"].sum()
    rows.append(
        {
            "equipment_id": equipment_id,
            "equipment_type": group["equipment_type"].iloc[0],
            "site_id": group["site_id"].iloc[0],
            "hours_since_last_service": round(float(hours_since), 1),
            "status": "Preventive due" if hours_since > SERVICE_INTERVAL_HOURS else "OK",
        }
    )

schedule_df = (
    pd.DataFrame(rows)
    .sort_values("hours_since_last_service", ascending=False)
    .reset_index(drop=True)
)

if schedule_df.empty:
    empty_state("No equipment in the current selection.")
else:
    due_count = int((schedule_df["status"] == "Preventive due").sum())
    if due_count:
        st.warning(f"{due_count} unit(s) exceed {SERVICE_INTERVAL_HOURS} operating hours since last service.")
    st.dataframe(
        schedule_df,
        width="stretch",
        hide_index=True,
        column_config={
            "hours_since_last_service": st.column_config.NumberColumn(
                "Hours since last service", format="%.1f"
            ),
        },
    )
    st.caption(
        f"Service interval: {SERVICE_INTERVAL_HOURS} operating hours. "
        "Units never serviced count hours from the start of the selected period."
    )

# ---------------------------------------------------------------------------
# Section 4: Failure risk prediction (ML, 7-day horizon)
# ---------------------------------------------------------------------------
st.subheader("Failure risk prediction (7-day horizon)")

failure_bundle = _failure_bundle()
if failure_bundle is None:
    st.info(
        "The failure-prediction model has not been trained yet, so this page "
        "shows the rule-based schedule only. To enable risk predictions, run: "
        "`python scripts/train_all.py`"
    )
else:
    failure_model, failure_features = _model_and_features(failure_bundle)
    failure_frame = make_failure_frame(df)
    failure_frame = failure_frame.assign(
        failure_probability=failure_model.predict_proba(
            failure_frame[failure_features]
        )[:, 1]
    )
    latest_risk = (
        failure_frame.loc[
            failure_frame.groupby("equipment_id")["date"].idxmax()
        ]
        .sort_values("failure_probability", ascending=False)
        .reset_index(drop=True)
    )
    latest_risk["failure_probability_pct"] = latest_risk["failure_probability"] * 100.0

    st.dataframe(
        latest_risk[
            [
                "equipment_id",
                "equipment_type",
                "site_id",
                "failure_probability_pct",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "equipment_id": "Equipment",
            "equipment_type": "Type",
            "site_id": "Site",
            "failure_probability_pct": st.column_config.NumberColumn(
                "P(failure in 7 days)", format="%.1f%%"
            ),
        },
    )

    fig_risk = px.bar(
        latest_risk,
        x="equipment_id",
        y="failure_probability",
        color="equipment_type",
        title="Predicted 7-day failure probability per equipment",
        labels={
            "equipment_id": "Equipment",
            "failure_probability": "P(failure in 7 days)",
            "equipment_type": "Type",
        },
    )
    fig_risk.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_risk, width="stretch")
    st.caption(
        "Predicted on the most recent day per equipment in the current "
        "selection. Higher bars = schedule inspection/maintenance sooner."
    )

    # --- SHAP explanation for one selected unit ---
    st.markdown("**Why is this unit at risk? (SHAP)**")
    explain_unit = st.selectbox(
        "Equipment to explain",
        options=sorted(latest_risk["equipment_id"].unique().tolist()),
        key="failure_explain_unit",
    )
    explain_row = latest_risk.loc[
        latest_risk["equipment_id"] == explain_unit
    ].iloc[0]
    shap_background = failure_frame[failure_features].sample(
        n=min(200, len(failure_frame)), random_state=42
    )
    explanation = explain_failure_risk(
        failure_model,
        explain_row[failure_features],
        failure_features,
        shap_background,
    )
    if explanation is None:
        st.caption("SHAP explanation skipped (the optional `shap` package is not installed).")
    else:
        contrib_names, contrib_values = explanation
        order = np.argsort(np.abs(contrib_values))[::-1]
        contrib_df = pd.DataFrame(
            {
                "feature": [contrib_names[i] for i in order],
                "shap_value": [contrib_values[i] for i in order],
            }
        )
        fig_shap = px.bar(
            contrib_df,
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="RdBu_r",
            title=f"SHAP contributions to failure risk — {explain_unit}",
            labels={"shap_value": "SHAP value", "feature": "Feature"},
        )
        st.plotly_chart(fig_shap, width="stretch")
        st.caption(
            "Positive values push the predicted failure probability up; "
            "negative values push it down."
        )

# ---------------------------------------------------------------------------
# Section 5: KPI metrics row
# ---------------------------------------------------------------------------
st.subheader("Key metrics")

total_cost = float(history["maintenance_cost_usd"].sum()) if not history.empty else 0.0

corrective_cost = (
    float(history.loc[history["maintenance_type"] == "corrective", "maintenance_cost_usd"].sum())
    if not history.empty
    else 0.0
)
corrective_share = (corrective_cost / total_cost * 100) if total_cost > 0 else 0.0

# Mean time between corrective events per equipment (days), averaged over units.
days_in_range = 0
if not df.empty:
    date_min = pd.to_datetime(df["date"]).min()
    date_max = pd.to_datetime(df["date"]).max()
    days_in_range = (date_max - date_min).days + 1

corrective_events = history[history["maintenance_type"] == "corrective"] if not history.empty else history
corrective_counts = corrective_events.groupby("equipment_id").size() if not corrective_events.empty else pd.Series(dtype=float)
units_with_corrective = int((corrective_counts > 0).sum()) if not corrective_counts.empty else 0

if units_with_corrective and days_in_range > 0:
    mtbc_per_unit = days_in_range / corrective_counts[corrective_counts > 0]
    mean_mtbc = float(mtbc_per_unit.mean())
else:
    mean_mtbc = 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Total maintenance cost (USD)", f"${total_cost:,.0f}")
col2.metric("Corrective share of cost", f"{corrective_share:.1f}%")
col3.metric("Mean time between corrective (days/unit)", f"{mean_mtbc:.1f}")
