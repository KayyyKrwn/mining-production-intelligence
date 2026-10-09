"""Unit tests for src/features.py.

Task 1 (Fase 2 ML plan): verify leakage-free feature engineering.
"""
import pandas as pd
import pytest

from features import (
    make_anomaly_frame,
    make_cost_frame,
    make_failure_frame,
    make_forecast_frame,
    make_ore_frame,
)

_OPS_DEFAULTS = {
    "site_id": "SITE-A",
    "equipment_id": "TRK-001",
    "equipment_type": "haul_truck",
    "age_years": 5,
    "operating_hours": 10.0,
    "production_ton": 100.0,
    "fuel_liter": 500.0,
    "downtime_hours": 0.0,
    "downtime_reason": "planned",
    "maintenance_type": "none",
    "maintenance_cost_usd": 0.0,
    "workers_on_shift": 20,
    "temp_c": 28.0,
    "rainfall_mm": 0.0,
    "ore_grade_pct": 58.0,
    "operating_cost_usd": 1000.0,
    "target_ton": 12000.0,
}


def _ops_frame(rows: list[dict]) -> pd.DataFrame:
    merged = [{**_OPS_DEFAULTS, **r} for r in rows]
    df = pd.DataFrame(merged)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_forecast_frame_lag_no_leakage():
    """lag_1 of the last row must be the previous day's value (40), not 50."""
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    # Pad the front so lag_30 rows survive the drop; the synthetic
    # series [10,20,30,40,50] occupies the last five days.
    production = [0.0] * 30 + [10.0, 20.0, 30.0, 40.0, 50.0]
    df = _ops_frame(
        [{"date": d, "production_ton": p} for d, p in zip(dates, production)]
    )
    frame = make_forecast_frame(df)
    assert len(frame) > 0
    last = frame.iloc[-1]
    assert last["production_ton"] == 50.0
    assert last["lag_1"] == 40.0
    assert last["lag_7"] == 0.0
    assert frame.filter(like="lag_").notna().all().all()


def test_failure_frame_labels_no_leakage():
    """A mechanical_failure on day 10 labels days 3-9 as 1, day 10+ as 0,
    and never leaks across equipment units."""
    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    rows_a = [
        {
            "date": d,
            "equipment_id": "EQ-A",
            "downtime_reason": "mechanical_failure" if i == 9 else "planned",
        }
        for i, d in enumerate(dates)
    ]
    rows_b = [
        {"date": d, "equipment_id": "EQ-B", "downtime_reason": "planned"}
        for d in dates
    ]
    frame = make_failure_frame(_ops_frame(rows_a + rows_b))
    a = (
        frame[frame["equipment_id"] == "EQ-A"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    b = (
        frame[frame["equipment_id"] == "EQ-B"]
        .sort_values("date")
        .reset_index(drop=True)
    )
    assert a["failure_next_7d"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1] + [0] * 6
    assert b["failure_next_7d"].tolist() == [0] * 15


def test_ore_frame_aggregates_one_hour():
    """Three rows within the same hour collapse into a single hourly row."""
    base = pd.Timestamp("2024-01-01 08:00")
    df = pd.DataFrame(
        [
            {
                "date": base + pd.Timedelta(minutes=15 * i),
                "site_id": "SITE-A",
                "silica_pct": v,
                "iron_feed_pct": 65.0,
                "reagent_flow": 10.0,
                "air_flow": 200.0,
                "ph": 9.5,
                "density": 2.6,
            }
            for i, v in enumerate([1.0, 2.0, 3.0])
        ]
    )
    frame = make_ore_frame(df)
    assert len(frame) == 1
    assert frame.iloc[0]["silica_pct"] == pytest.approx(2.0)
    assert frame.iloc[0]["iron_feed_pct"] == pytest.approx(65.0)


def test_cost_frame_site_day_sums():
    """make_cost_frame sums the day across units of the same site."""
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    rows = [
        {
            "date": d,
            "equipment_id": eq,
            "operating_cost_usd": 1000.0,
            "maintenance_type": mt,
        }
        for d in dates
        for eq, mt in [("EQ-A", "corrective"), ("EQ-B", "none")]
    ]
    frame = make_cost_frame(_ops_frame(rows))
    assert len(frame) == 2
    assert frame["operating_cost_usd"].tolist() == [2000.0, 2000.0]
    assert frame["n_corrective"].tolist() == [1, 1]
    assert frame["n_preventive"].tolist() == [0, 0]


def test_anomaly_frame_ratios():
    """make_anomaly_frame returns per-unit-day ratios without NaN leakage."""
    df = _ops_frame(
        [
            {"date": "2024-01-01", "equipment_id": "EQ-A"},
            {"date": "2024-01-02", "equipment_id": "EQ-A"},
        ]
    )
    frame = make_anomaly_frame(df)
    assert len(frame) == 2
    assert frame["fuel_per_ton"].tolist() == [5.0, 5.0]
    assert frame["cost_per_ton"].tolist() == [10.0, 10.0]
    assert frame.filter(like="_per_ton").notna().all().all()
