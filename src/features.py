"""Feature engineering for the Mining Production & Predictive Maintenance models.

Pure functions: each takes a DataFrame produced by ``data_loader`` and
returns a modelling-ready frame. No temporal leakage: a feature for day ``t``
only uses data up to and including ``t`` (all lags are explicitly shifted by
at least one step, failure labels look only at strictly future days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division that maps 0/0 and x/0 to 0.0 instead of NaN/inf."""
    return (num / den.replace(0, np.nan)).fillna(0.0)


def _slope_14d(s: pd.Series) -> pd.Series:
    """Slope of a linear fit over the last 14 values (0.0 for short series)."""

    def _slope(window: np.ndarray) -> float:
        if len(window) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(window)), window, 1)[0])

    return s.rolling(14, min_periods=1).apply(_slope, raw=True).fillna(0.0)


def _hours_since_maintenance(group: pd.DataFrame) -> pd.Series:
    """Hours (days*24) since the last maintenance day; 0 on maintenance days."""
    group = group.sort_values("date")
    maint = (group["maintenance_type"] != "none").to_numpy()
    idx = np.arange(len(group))
    last = np.maximum.accumulate(np.where(maint, idx, -1))
    days = np.where(last >= 0, idx - last, idx + 1)
    return pd.Series(days * 24.0, index=group.index)


def make_forecast_frame(ops: pd.DataFrame) -> pd.DataFrame:
    """Build a site-day frame for production forecasting.

    Aggregates ``production_ton`` (sum), ``temp_c`` (mean), ``rainfall_mm``
    (sum) and the active unit count per site-day, then adds ``lag_1/7/14/30``,
    ``roll_mean_7/30`` (computed on values strictly before day ``t``),
    ``dow`` and ``month``. Rows whose lag features are NaN are dropped.
    """
    ops = ops.copy()
    ops["date"] = pd.to_datetime(ops["date"])
    daily = (
        ops.groupby(["site_id", "date"], as_index=False)
        .agg(
            production_ton=("production_ton", "sum"),
            temp_c=("temp_c", "mean"),
            rainfall_mm=("rainfall_mm", "sum"),
            n_units=("equipment_id", "nunique"),
        )
        .sort_values(["site_id", "date"])
    )
    by_site = daily.groupby("site_id")["production_ton"]
    for k in (1, 7, 14, 30):
        daily[f"lag_{k}"] = by_site.shift(k).to_numpy()
    daily["roll_mean_7"] = by_site.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    ).to_numpy()
    daily["roll_mean_30"] = by_site.transform(
        lambda s: s.shift(1).rolling(30, min_periods=1).mean()
    ).to_numpy()
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    lag_cols = [c for c in daily.columns if c.startswith("lag_")]
    return daily.dropna(subset=lag_cols).reset_index(drop=True)


def make_failure_frame(ops: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Build an equipment-day frame for failure prediction.

    ``failure_next_7d`` is 1 when a ``mechanical_failure`` occurs on any of
    the ``horizon`` strictly-future days (computed per equipment, so labels
    never leak across units). Degradation features (fuel slope, recent minor
    downtime, hours since maintenance) only use data up to day ``t``.
    """
    ops = ops.copy()
    ops["date"] = pd.to_datetime(ops["date"])
    ops = ops.sort_values(["equipment_id", "date"]).reset_index(drop=True)
    by_unit = ops.groupby("equipment_id")

    ops["fuel_per_ton"] = _safe_div(ops["fuel_liter"], ops["production_ton"])
    ops["fuel_trend_14d"] = by_unit["fuel_liter"].transform(_slope_14d).to_numpy()
    minor = (ops["downtime_reason"] == "mechanical_minor").astype(int)
    ops["minor_downtime_14d"] = (
        minor.groupby(ops["equipment_id"])
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).sum())
        .fillna(0.0)
        .to_numpy()
    )
    ops["hours_since_maintenance"] = by_unit.apply(
        _hours_since_maintenance, include_groups=False
    ).reset_index(level=0, drop=True).to_numpy()

    failure = (ops["downtime_reason"] == "mechanical_failure").astype(int)
    label = pd.Series(np.zeros(len(ops), dtype=int), index=ops.index)
    for k in range(1, horizon + 1):
        label = label | failure.groupby(ops["equipment_id"]).shift(-k).fillna(0).astype(int)
    ops["failure_next_7d"] = label.astype(int)

    return ops[
        [
            "date",
            "site_id",
            "equipment_id",
            "equipment_type",
            "failure_next_7d",
            "fuel_per_ton",
            "fuel_trend_14d",
            "minor_downtime_14d",
            "hours_since_maintenance",
            "age_years",
            "operating_hours",
        ]
    ]


def make_cost_frame(ops: pd.DataFrame) -> pd.DataFrame:
    """Build a site-day frame for operating-cost prediction.

    The target is ``operating_cost_usd`` (sum); features are production, fuel,
    downtime and maintenance aggregates for the same site-day.
    """
    ops = ops.copy()
    ops["date"] = pd.to_datetime(ops["date"])
    daily = (
        ops.groupby(["site_id", "date"], as_index=False)
        .agg(
            operating_cost_usd=("operating_cost_usd", "sum"),
            production_ton=("production_ton", "sum"),
            fuel_liter=("fuel_liter", "sum"),
            downtime_hours=("downtime_hours", "sum"),
            n_corrective=(
                "maintenance_type",
                lambda s: int((s == "corrective").sum()),
            ),
            n_preventive=(
                "maintenance_type",
                lambda s: int((s == "preventive").sum()),
            ),
            maintenance_cost_usd=("maintenance_cost_usd", "sum"),
            workers_on_shift=("workers_on_shift", "sum"),
            rainfall_mm=("rainfall_mm", "sum"),
        )
        .sort_values(["site_id", "date"])
        .reset_index(drop=True)
    )
    return daily


def make_ore_frame(flot: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an ore-quality frame to hourly rows (numeric mean).

    ``silica_pct`` is kept as the modelling target. ``site_id`` is preserved
    as a group key when present (synthetic fallback); the Kaggle frame has
    none and is aggregated plant-wide.
    """
    flot = flot.copy()
    flot["date"] = pd.to_datetime(flot["date"])
    flot["hour"] = flot["date"].dt.floor("h")
    keys = ["hour"] + (["site_id"] if "site_id" in flot.columns else [])
    num_cols = flot.select_dtypes(include="number").columns.tolist()
    hourly = (
        flot.groupby(keys, as_index=False)[num_cols]
        .mean()
        .rename(columns={"hour": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return hourly


def make_anomaly_frame(ops: pd.DataFrame) -> pd.DataFrame:
    """Build an equipment-day frame of efficiency ratios for anomaly detection."""
    ops = ops.copy()
    ops["date"] = pd.to_datetime(ops["date"])
    ops["fuel_per_ton"] = _safe_div(ops["fuel_liter"], ops["production_ton"])
    ops["cost_per_ton"] = _safe_div(ops["operating_cost_usd"], ops["production_ton"])
    return ops[
        [
            "date",
            "site_id",
            "equipment_id",
            "equipment_type",
            "fuel_per_ton",
            "cost_per_ton",
            "downtime_hours",
            "operating_hours",
        ]
    ].sort_values(["equipment_id", "date"]).reset_index(drop=True)
