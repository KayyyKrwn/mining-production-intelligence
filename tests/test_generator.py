"""Tests for src/data_generator.py (Task 2: data generator).

Written FIRST from the task spec (TDD). Covers:
(a) reproducibility: two generate(42) calls produce identical DataFrames (hash equal)
(b) value ranges: fuel_liter >= 0, production_ton >= 0, 0 <= downtime_hours <= 24,
    mechanical_failure share within 1.0% - 2.5%
(c) exactly 3 sites and 40 unique equipment_ids
plus schema, fleet composition, cost consistency, and the degradation-signal
contract that Phase-2 ML depends on.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate

OPS_COLUMNS = [
    "date",
    "site_id",
    "equipment_id",
    "equipment_type",
    "age_years",
    "operating_hours",
    "production_ton",
    "fuel_liter",
    "downtime_hours",
    "downtime_reason",
    "maintenance_type",
    "maintenance_cost_usd",
    "workers_on_shift",
    "temp_c",
    "rainfall_mm",
    "ore_grade_pct",
    "operating_cost_usd",
    "target_ton",
]

ORE_COLUMNS = [
    "date",
    "site_id",
    "silica_pct",
    "iron_feed_pct",
    "reagent_flow",
    "air_flow",
    "ph",
    "density",
]

N_DAYS = 731  # 2024-01-01 .. 2025-12-31 (2024 is a leap year)
N_EQUIPMENT = 40
N_SITES = 3


@pytest.fixture(scope="module")
def frames():
    return generate(42)


def _hash(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=True)


# ---------------------------------------------------------------- (a) reproducibility
def test_reproducible_same_seed(frames):
    ops1, ore1 = frames
    ops2, ore2 = generate(42)
    assert ops1.equals(ops2)
    assert ore1.equals(ore2)
    assert _hash(ops1).equals(_hash(ops2))
    assert _hash(ore1).equals(_hash(ore2))


def test_different_seed_differs():
    ops_a, _ = generate(42)
    ops_b, _ = generate(43)
    assert not ops_a.equals(ops_b)


# ---------------------------------------------------------------- schema
def test_ops_schema(frames):
    ops, _ = frames
    assert list(ops.columns) == OPS_COLUMNS
    assert len(ops) == N_EQUIPMENT * N_DAYS
    assert ops["date"].min() == pd.Timestamp("2024-01-01")
    assert ops["date"].max() == pd.Timestamp("2025-12-31")
    assert ops["date"].nunique() == N_DAYS
    assert ops.isna().sum().sum() == 0


def test_ore_schema(frames):
    _, ore = frames
    assert list(ore.columns) == ORE_COLUMNS
    assert len(ore) == N_SITES * N_DAYS
    assert ore.isna().sum().sum() == 0
    # silica base range per spec (2.5-4.5) with modest correlation-driven spread
    assert ore["silica_pct"].between(0.5, 7.0).all()


def test_categorical_domains(frames):
    ops, _ = frames
    assert set(ops["downtime_reason"].unique()) <= {
        "mechanical_minor",
        "mechanical_failure",
        "weather",
        "safety",
        "planned",
    }
    assert set(ops["maintenance_type"].unique()) <= {
        "none",
        "routine",
        "preventive",
        "corrective",
    }
    assert set(ops["equipment_type"].unique()) == {
        "haul_truck",
        "excavator",
        "drill",
        "loader",
    }


# ---------------------------------------------------------------- (b) value ranges
def test_value_ranges(frames):
    ops, _ = frames
    assert (ops["fuel_liter"] >= 0).all()
    assert (ops["production_ton"] >= 0).all()
    assert (ops["downtime_hours"] >= 0).all()
    assert (ops["downtime_hours"] <= 24).all()
    fail_rate = (ops["downtime_reason"] == "mechanical_failure").mean()
    assert 0.010 <= fail_rate <= 0.025, f"failure rate {fail_rate:.4f} out of bounds"


def test_age_range(frames):
    ops, _ = frames
    assert ops["age_years"].between(1, 12).all()


# ---------------------------------------------------------------- (c) sites & fleet
def test_sites_and_equipment(frames):
    ops, _ = frames
    assert sorted(ops["site_id"].unique().tolist()) == ["SITE-A", "SITE-B", "SITE-C"]
    assert ops["equipment_id"].nunique() == N_EQUIPMENT


def test_fleet_composition(frames):
    ops, _ = frames
    per_type = ops.drop_duplicates("equipment_id")["equipment_type"].value_counts()
    assert per_type["haul_truck"] == 16
    assert per_type["excavator"] == 8
    assert per_type["drill"] == 8
    assert per_type["loader"] == 8
    # ids look like HT-001 and each unit is pinned to a single site
    assert ops["equipment_id"].str.match(r"^[A-Z]{2}-\d{3}$").all()
    assert (ops.groupby("equipment_id")["site_id"].nunique() == 1).all()


def test_site_targets(frames):
    ops, _ = frames
    medians = ops.groupby("site_id")["target_ton"].median()
    assert medians["SITE-A"] == pytest.approx(12000, rel=0.05)
    assert medians["SITE-B"] == pytest.approx(8000, rel=0.05)
    assert medians["SITE-C"] == pytest.approx(5000, rel=0.05)


# ---------------------------------------------------------------- cost consistency
def test_maintenance_costs(frames):
    ops, _ = frames
    costs = ops.groupby("maintenance_type")["maintenance_cost_usd"]
    assert (ops.loc[ops["maintenance_type"] == "none", "maintenance_cost_usd"] == 0).all()
    assert costs.get_group("routine").between(200, 800).all()
    assert costs.get_group("preventive").between(1500, 5000).all()
    assert costs.get_group("corrective").between(8000, 30000).all()
    # failures are exactly the corrective-maintenance rows
    fail = ops["downtime_reason"] == "mechanical_failure"
    assert (ops.loc[fail, "maintenance_type"] == "corrective").all()
    assert (~fail).sum() > 0 and (ops.loc[~fail, "maintenance_type"] != "corrective").all()


def test_operating_cost_formula(frames):
    ops, _ = frames
    expected = (
        ops["fuel_liter"] * 0.9
        + ops["workers_on_shift"] * 25
        + ops["maintenance_cost_usd"]
        + 500
    )
    assert np.allclose(ops["operating_cost_usd"], expected, atol=0.05)


# ---------------------------------------------------------------- degradation signal (Phase-2 ML contract)
def test_degradation_signal_before_failures(frames):
    """Fuel burn in the 14 days preceding a failure must be systematically
    higher than each unit's overall average (learnable predictive signal)."""
    ops, _ = frames
    ops = ops.sort_values(["equipment_id", "date"]).reset_index(drop=True)
    fail_idx = ops.index[ops["downtime_reason"] == "mechanical_failure"]
    assert len(fail_idx) > 100  # enough events to learn from
    window_fuel, base_fuel = [], []
    for i in fail_idx:
        eq = ops.at[i, "equipment_id"]
        unit = ops[ops["equipment_id"] == eq]
        pos = unit.index.get_loc(i)
        window = unit.iloc[max(0, pos - 14) : pos]
        if len(window) >= 7:
            window_fuel.append(window["fuel_liter"].mean())
            base_fuel.append(unit["fuel_liter"].mean())
    assert np.mean(window_fuel) > np.mean(base_fuel) * 1.05
