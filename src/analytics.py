"""Analytics KPIs for the Mining Production & Predictive Maintenance
Intelligence dashboard.

Pure functions over already-filtered DataFrames. None of them ever raise
ZeroDivisionError: a zero denominator yields 0.0.
"""
from __future__ import annotations

import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 instead of raising on a zero denominator."""
    return numerator / denominator if denominator else 0.0


def production_efficiency(df: pd.DataFrame) -> float:
    """Actual production vs target, in percent."""
    return _safe_div(df["production_ton"].sum(), df["target_ton"].sum()) * 100.0


def cost_per_ton(df: pd.DataFrame) -> float:
    """Operating cost (USD) per ton produced."""
    return _safe_div(df["operating_cost_usd"].sum(), df["production_ton"].sum())


def equipment_utilization(df: pd.DataFrame) -> float:
    """Share of available time the equipment was operating, in percent."""
    total_hours = (df["operating_hours"] + df["downtime_hours"]).sum()
    return _safe_div(df["operating_hours"].sum(), total_hours) * 100.0


def fuel_efficiency(df: pd.DataFrame) -> float:
    """Liters of fuel consumed per ton produced (lower is better)."""
    return _safe_div(df["fuel_liter"].sum(), df["production_ton"].sum())


def downtime_summary(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """Downtime KPIs.

    Returns (downtime_percent, breakdown): downtime as a percent of total
    available hours, plus a Series of each downtime_reason's share of total
    downtime hours, in percent (sums to 100; empty Series when no downtime).
    """
    total_downtime = float(df["downtime_hours"].sum())
    total_hours = float((df["operating_hours"] + df["downtime_hours"]).sum())
    pct = _safe_div(total_downtime, total_hours) * 100.0
    if total_downtime:
        breakdown = (
            df.groupby("downtime_reason")["downtime_hours"].sum()
            / total_downtime
            * 100.0
        )
    else:
        breakdown = pd.Series(dtype=float)
    return pct, breakdown


def productivity(df: pd.DataFrame) -> float:
    """Tons produced per worker-day on shift."""
    return _safe_div(df["production_ton"].sum(), df["workers_on_shift"].sum())
