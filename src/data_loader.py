"""CSV loaders for the Mining Production & Predictive Maintenance dashboard.

load_operations: reads the daily equipment-operations log and validates the
    schema.
load_ore_quality: reads the flotation-plant ore-quality CSV (real Kaggle
    file when present, synthetic fallback otherwise).
"""
from pathlib import Path

import pandas as pd

OPERATIONS_REQUIRED_COLUMNS = {
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
}

# Kaggle flotation dataset columns -> canonical English identifiers.
KAGGLE_COLUMN_MAP = {
    "% silica concentrate": "silica_pct",
    "% iron concentrate": "iron_conc_pct",
    "% iron feed": "iron_feed_pct",
    "% silica feed": "silica_feed_pct",
    "date": "date",
}


def _normalize_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=KAGGLE_COLUMN_MAP)
    return df


def load_operations(path: str) -> pd.DataFrame:
    """Read an operations CSV; parse `date` as datetime; validate schema."""
    df = pd.read_csv(Path(path), parse_dates=["date"])
    missing = sorted(OPERATIONS_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Operations CSV is missing required column(s): "
            + ", ".join(missing)
            + f". Path: {path}"
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_ore_quality(raw_path: str, fallback_path: str) -> tuple[pd.DataFrame, str]:
    """Load the ore-quality dataset.

    If ``raw_path`` (real Kaggle export) exists it is read with
    ``decimal=","`` (the real file uses comma as decimal separator) and its
    columns are normalized; returns ``(df, "Kaggle")``. Otherwise the
    synthetic ``fallback_path`` is loaded; returns ``(df, "Synthetic")``.

    The normalized frame always contains ``silica_pct``.
    """
    raw = Path(raw_path)
    if raw.exists():
        df = pd.read_csv(raw, decimal=",")
        df = _normalize_kaggle_columns(df)
        return df, "Kaggle"
    df = pd.read_csv(Path(fallback_path))
    return df, "Synthetic"
