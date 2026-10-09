"""Tests for src/analytics.py — 6 KPI functions (Task 4)."""
import pandas as pd
import pytest

from analytics import (
    production_efficiency,
    cost_per_ton,
    equipment_utilization,
    fuel_efficiency,
    downtime_summary,
    productivity,
)


def make_df(**kwargs):
    """Base handcrafted DataFrame; override any column via kwargs.

    All columns are normalized to the row count of the longest overridden
    column (default 2) by cycling the base values, so overrides never
    produce ragged columns.
    """
    base = dict(
        production_ton=[60.0, 40.0],
        target_ton=[50.0, 30.0],
        operating_cost_usd=[1000.0, 600.0],
        operating_hours=[10.0, 8.0],
        downtime_hours=[4.0, 2.0],
        downtime_reason=["mechanical", "electrical"],
        fuel_liter=[200.0, 100.0],
        workers_on_shift=[10, 10],
    )
    base.update(kwargs)
    override_lengths = [len(v) for v in kwargs.values() if isinstance(v, (list, tuple))]
    n = max(override_lengths) if override_lengths else 2

    def _to_n(values):
        vals = list(values)
        return (vals * ((n + len(vals) - 1) // len(vals)))[:n]

    return pd.DataFrame({col: _to_n(vals) for col, vals in base.items()})


# --- production_efficiency ---
def test_production_efficiency():
    df = make_df(production_ton=[100.0], target_ton=[80.0])
    assert production_efficiency(df) == 125.0


# --- zero-denominator guards (no raise) ---
def test_cost_per_ton_zero_production():
    df = make_df(production_ton=[0.0, 0.0], operating_cost_usd=[500.0, 500.0])
    assert cost_per_ton(df) == 0.0


def test_fuel_efficiency_zero_production():
    df = make_df(production_ton=[0.0, 0.0], fuel_liter=[50.0, 50.0])
    assert fuel_efficiency(df) == 0.0


# --- equipment_utilization ---
def test_equipment_utilization():
    df = make_df(operating_hours=[18.0], downtime_hours=[6.0])
    assert equipment_utilization(df) == 75.0


# --- downtime_summary ---
def test_downtime_summary():
    df = make_df(
        operating_hours=[18.0, 18.0],
        downtime_hours=[4.0, 2.0],
        downtime_reason=["mechanical", "mechanical"],
    )
    pct, breakdown = downtime_summary(df)
    # total downtime 6h / (36h operating + 6h downtime) * 100 = 14.285714...
    assert pct == pytest.approx(6.0 / 42.0 * 100)
    assert isinstance(breakdown, pd.Series)
    assert breakdown["mechanical"] == pytest.approx(100.0)

    df2 = make_df(
        operating_hours=[18.0, 18.0, 18.0, 18.0],
        downtime_hours=[12.0, 6.0, 4.0, 2.0],
        downtime_reason=["mechanical", "electrical", "mechanical", "electrical"],
    )
    pct2, breakdown2 = downtime_summary(df2)
    # 24h downtime / 96h total = 25.0%
    assert pct2 == 25.0
    # 24h total downtime: mechanical 16h (66.67%), electrical 8h (33.33%)
    assert breakdown2["mechanical"] == pytest.approx(16.0 / 24.0 * 100)
    assert breakdown2["electrical"] == pytest.approx(8.0 / 24.0 * 100)


# --- empty DataFrame (0 rows) ---
EMPTY = pd.DataFrame(
    {
        "production_ton": pd.Series(dtype=float),
        "target_ton": pd.Series(dtype=float),
        "operating_cost_usd": pd.Series(dtype=float),
        "operating_hours": pd.Series(dtype=float),
        "downtime_hours": pd.Series(dtype=float),
        "downtime_reason": pd.Series(dtype=str),
        "fuel_liter": pd.Series(dtype=float),
        "workers_on_shift": pd.Series(dtype=float),
    }
)


def test_empty_dataframe():
    assert production_efficiency(EMPTY) == 0.0
    assert cost_per_ton(EMPTY) == 0.0
    assert equipment_utilization(EMPTY) == 0.0
    assert fuel_efficiency(EMPTY) == 0.0
    assert productivity(EMPTY) == 0.0
    pct, breakdown = downtime_summary(EMPTY)
    assert pct == 0.0
    assert isinstance(breakdown, pd.Series)
    assert len(breakdown) == 0


# --- productivity ---
def test_productivity():
    df = make_df(production_ton=[100.0, 100.0], workers_on_shift=[10, 10])
    assert productivity(df) == 10.0
