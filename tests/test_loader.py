"""Tests for src/data_loader.py.

Fixtures are written to tmp_path (under /tmp) so they never depend on
other tasks' outputs.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_operations, load_ore_quality  # noqa: E402

OPERATIONS_COLUMNS = [
    "date",
    "site_id",
    "equipment_id",
    "equipment_type",
    "operating_hours",
    "production_ton",
    "fuel_liter",
    "downtime_hours",
    "downtime_reason",
    "maintenance_cost_usd",
    "workers_on_shift",
    "operating_cost_usd",
    "target_ton",
]


def _write_operations_csv(path: Path, drop: list[str] | None = None) -> Path:
    cols = [c for c in OPERATIONS_COLUMNS if not (drop and c in drop)]
    row = ["2024-03-15", "site_a", "eq_01", "excavator", 20.5, 1500.0, 900.0,
           3.5, "planned maintenance", 1200.0, 8, 45000.0, 1600.0]
    df = pd.DataFrame([[row[OPERATIONS_COLUMNS.index(c)] for c in cols]], columns=cols)
    df.to_csv(path, index=False)
    return path


# --- load_ore_quality -----------------------------------------------------

def test_ore_quality_missing_raw_falls_back_to_synthetic(tmp_path):
    """(a) nonexistent raw_path -> label 'Synthetic', fallback df loads."""
    fallback = tmp_path / "fallback.csv"
    pd.DataFrame({"date": ["2024-01-01"], "silica_pct": [2.5]}).to_csv(fallback, index=False)

    df, label = load_ore_quality(str(tmp_path / "does_not_exist.csv"), str(fallback))

    assert label == "Synthetic"
    assert isinstance(df, pd.DataFrame)
    assert "silica_pct" in df.columns
    assert len(df) == 1


def test_ore_quality_kaggle_sample_normalized(tmp_path):
    """(b) raw Kaggle-style CSV with '% Silica Concentrate' -> silica_pct numeric."""
    raw = tmp_path / "raw_sample.csv"
    raw.write_text(
        "date,% Silica Concentrate,% Iron Concentrate\n"
        '2024-01-01 00:00:00,"1,31","66,5"\n'
        '2024-01-01 01:00:00,"2,72","68,0"\n',
        encoding="utf-8",
    )

    df, label = load_ore_quality(str(raw), str(tmp_path / "fallback.csv"))

    assert label == "Kaggle"
    assert "silica_pct" in df.columns
    assert "iron_conc_pct" in df.columns
    assert pd.api.types.is_numeric_dtype(df["silica_pct"])
    assert df["silica_pct"].iloc[0] == pytest.approx(1.31)
    assert df["iron_conc_pct"].iloc[1] == pytest.approx(68.0)


# --- load_operations ------------------------------------------------------

def test_load_operations_parses_date_as_datetime(tmp_path):
    """(c) load_operations parses `date` as datetime64."""
    csv_path = _write_operations_csv(tmp_path / "ops.csv")

    df = load_operations(str(csv_path))

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].iloc[0] == pd.Timestamp("2024-03-15")


def test_load_operations_raises_value_error_naming_missing_column(tmp_path):
    """(d) missing required column -> ValueError naming the missing column."""
    csv_path = _write_operations_csv(tmp_path / "ops_bad.csv", drop=["target_ton"])

    with pytest.raises(ValueError, match="target_ton"):
        load_operations(str(csv_path))
