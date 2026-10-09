"""Synthetic mining-operations data generator.

Generates realistic, reproducible open-pit mining operations data for three
sites over 2024-01-01 .. 2025-12-31 (731 days) across 40 equipment units.

Public API:
    generate(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]
        Returns (ops_df, ore_df):
        - ops_df: one row per equipment x day (40 x 731 = 29,240 rows)
        - ore_df: one row per site x day (3 x 731 = 2,193 rows), a synthetic
          fallback for the real Kaggle flotation dataset.

CLI:
    python src/data_generator.py [--seed 42] [--out-dir data]
        Writes data/mining_operations.csv and data/ore_quality_synthetic.csv.

Design notes:
- Determinism: a single numpy.random.Generator is created from ``seed`` and
  every random draw happens in a fixed order. No dict/set iteration or
  PYTHONHASHSEED-dependent hashing is used, so two calls with the same seed
  produce bit-identical DataFrames on any platform.
- Failure patterns carry learnable signal for predictive-maintenance models:
  ~1.5% of equipment-days are mechanical failures (older units fail more
  often); in the 7-14 days before each failure, fuel burn rises 15-30% and
  minor mechanical issues become 3x more likely.
- numpy/pandas only; everything is vectorised over equipment x day.

The data is synthetic but calibrated against public industry benchmarks
(typical open-pit truck/excavator/drill/loader utilisation, fuel burn and
maintenance cost bands). It is clearly labelled synthetic in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------- constants
SITES = ("SITE-A", "SITE-B", "SITE-C")
SITE_TARGETS = {"SITE-A": 12_000.0, "SITE-B": 8_000.0, "SITE-C": 5_000.0}
SITE_WORKERS = {"SITE-A": 60, "SITE-B": 40, "SITE-C": 25}

# (equipment_type, id_prefix, count, units per site [A, B, C])
FLEET_PLAN = (
    ("haul_truck", "HT", 16, (8, 5, 3)),
    ("excavator", "EX", 8, (4, 2, 2)),
    ("drill", "DR", 8, (3, 3, 2)),
    ("loader", "LO", 8, (4, 2, 2)),
)

# Per-type operating parameters: base daily hours, tonnes per operating hour,
# litres of fuel per operating hour.
TYPE_PARAMS = {
    "haul_truck": {"base_hours": 20.0, "ton_per_hour": 320.0, "liter_per_hour": 300.0},
    "excavator": {"base_hours": 18.0, "ton_per_hour": 450.0, "liter_per_hour": 180.0},
    "drill": {"base_hours": 14.0, "ton_per_hour": 60.0, "liter_per_hour": 90.0},
    "loader": {"base_hours": 16.0, "ton_per_hour": 280.0, "liter_per_hour": 120.0},
}

FAILURE_RATE = 0.015  # ~1.5% of equipment-days are mechanical failures
RAINY_MONTHS = {11, 12, 1, 2, 3}  # Nov-Mar rainy season
FUEL_PRICE_USD = 0.9
LABOUR_COST_USD_PER_WORKER = 25.0
OVERHEAD_USD = 500.0

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


# ---------------------------------------------------------------- roster
def _build_roster(rng: np.random.Generator) -> pd.DataFrame:
    """Deterministic equipment roster: stable ids, type, site, age (1-12)."""
    rows: list[dict] = []
    for eq_type, prefix, count, split in FLEET_PLAN:
        assert sum(split) == count
        bounds = np.cumsum((0,) + split)  # e.g. (0, 8, 13, 16)
        for i in range(count):
            site_pos = int(np.searchsorted(bounds, i, side="right") - 1)
            rows.append(
                {
                    "equipment_id": f"{prefix}-{i + 1:03d}",
                    "equipment_type": eq_type,
                    "site_id": SITES[site_pos],
                }
            )
    roster = pd.DataFrame(rows)
    roster["age_years"] = rng.integers(1, 13, size=len(roster))  # 1..12 inclusive
    return roster


# ---------------------------------------------------------------- main generator
def generate(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    roster = _build_roster(rng)
    n_eq = len(roster)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    n_days = len(dates)
    n = n_eq * n_days

    site_of_eq = roster["site_id"].map({s: i for i, s in enumerate(SITES)}).to_numpy()
    type_of_eq = roster["equipment_type"].to_numpy()
    age_of_eq = roster["age_years"].to_numpy().astype(float)

    # Equipment-major layout: rows for unit i live in [i*n_days, (i+1)*n_days).
    eq_idx = np.repeat(np.arange(n_eq), n_days)
    day_idx = np.tile(np.arange(n_days), n_eq)
    site_idx = site_of_eq[eq_idx]
    age = age_of_eq[eq_idx]

    # ------------------------------------------------ site x day weather & site state
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()
    seasonal = np.sin(2 * np.pi * (doy - 15) / 365.0)

    temp_site = 26.0 + 3.0 * seasonal[:, None] + rng.normal(0, 1.0, (n_days, 3))

    rainy = np.isin(month, list(RAINY_MONTHS))[:, None]
    p_rain = np.where(rainy, 0.55, 0.08)
    rain_intensity = np.where(
        rainy,
        rng.exponential(12.0, (n_days, 3)).clip(0, 120),
        rng.exponential(4.0, (n_days, 3)).clip(0, 30),
    )
    rainfall_site = np.where(rng.random((n_days, 3)) < p_rain, rain_intensity, 0.0)

    workers_base = np.array([SITE_WORKERS[s] for s in SITES])
    workers_site = np.rint(
        workers_base[None, :] + rng.normal(0, 2.5, (n_days, 3))
    ).clip(10, None)

    grade_base = rng.uniform(58.0, 65.0, size=3)  # per-site ore grade base
    grade_site = (grade_base[None, :] + rng.normal(0, 1.2, (n_days, 3))).clip(52, 68)

    target_base = np.array([SITE_TARGETS[s] for s in SITES])
    target_site = target_base[None, :] * rng.uniform(0.97, 1.03, (n_days, 3))

    # ------------------------------------------------ per equipment-day rows
    rainfall = rainfall_site[day_idx, site_idx]
    temp = temp_site[day_idx, site_idx]
    workers = workers_site[day_idx, site_idx].astype(int)
    ore_grade = grade_site[day_idx, site_idx]
    target_ton = target_site[day_idx, site_idx]

    base_hours = np.array([TYPE_PARAMS[t]["base_hours"] for t in type_of_eq[eq_idx]])
    ton_per_hour = np.array([TYPE_PARAMS[t]["ton_per_hour"] for t in type_of_eq[eq_idx]])
    liter_per_hour = np.array([TYPE_PARAMS[t]["liter_per_hour"] for t in type_of_eq[eq_idx]])

    # Operating hours: type base +/- noise, reduced by heavy rain (rainy season).
    rain_reduction = np.minimum(rainfall * 0.08, 6.0)
    operating_hours = np.clip(
        base_hours + rng.normal(0, 1.2, n) - rain_reduction, 0.0, 22.0
    )

    # Mechanical failures: ~1.5% of equipment-days; older units fail more often.
    p_fail = FAILURE_RATE * age / age_of_eq.mean()
    is_failure = rng.random(n) < p_fail

    # Maintenance: failures are always corrective; otherwise routine/preventive
    # draws (older equipment serviced slightly more often), else none.
    u_maint = rng.random(n)
    p_routine = 0.06 + 0.004 * age
    is_preventive = ~is_failure & (u_maint < 0.02)
    is_routine = ~is_failure & ~is_preventive & (u_maint < 0.02 + p_routine)
    maintenance_type = np.select(
        [is_failure, is_preventive, is_routine],
        ["corrective", "preventive", "routine"],
        default="none",
    )

    # Planned maintenance hours consume part of the 24h day.
    maint_hours = np.select(
        [is_failure, is_preventive, is_routine],
        [
            rng.uniform(8, 16, n),  # corrective: major repair window
            rng.uniform(4, 8, n),  # preventive: scheduled service
            rng.uniform(1, 3, n),  # routine: inspection / small fix
        ],
        default=0.0,
    )
    # Maintenance and failures eat into operating time.
    operating_hours = np.minimum(operating_hours, 24.0 - maint_hours)
    operating_hours = np.where(
        is_failure,
        rng.uniform(0.5, 4.0, n),  # breakdown part-way through the shift
        operating_hours,
    )
    operating_hours = np.clip(operating_hours, 0.0, 22.0)

    # ------------------------------------------------ degradation before failures
    # The 7-14 days before each failure: fuel burn x1.15-1.3 and 3x higher
    # chance of a minor mechanical issue. This is the learnable signal.
    degrade = np.zeros(n, dtype=bool)
    fuel_mult = np.ones(n)
    for i in range(n_eq):
        sl = slice(i * n_days, (i + 1) * n_days)
        for fd in np.flatnonzero(is_failure[sl]):
            w = int(rng.integers(7, 15))  # 7..14 days before the failure
            m = float(rng.uniform(1.15, 1.30))
            lo = max(0, fd - w)
            seg_mask = degrade[sl]
            seg_mask[lo:fd] = True  # basic slice -> view, assignment sticks
            seg_mult = fuel_mult[sl]
            np.maximum(seg_mult[lo:fd], m, out=seg_mult[lo:fd])

    # Minor mechanical issues: 1.5% base rate, 4.5% inside degradation windows.
    is_minor = rng.random(n) < np.where(degrade, 0.045, 0.015)
    is_safety = rng.random(n) < 0.003
    operating_hours = np.where(is_minor & ~is_failure, operating_hours * 0.9, operating_hours)
    operating_hours = np.where(is_safety & ~is_failure, operating_hours * 0.7, operating_hours)
    operating_hours = np.clip(operating_hours, 0.0, 22.0)

    # Downtime is whatever is left of the 24h day (spec formula).
    downtime_hours = np.maximum(0.0, 24.0 - operating_hours - maint_hours)

    # Downtime reason by priority. Ordinary idle time (shift gaps between the
    # ~14-20 operating hours and a full day) is labelled "planned".
    downtime_reason = np.select(
        [
            is_failure,
            is_preventive | is_routine,
            rainfall > 25.0,
            is_safety,
            is_minor,
        ],
        ["mechanical_failure", "planned", "weather", "safety", "mechanical_minor"],
        default="planned",
    )

    # Production, fuel, costs.
    production_ton = operating_hours * ton_per_hour * rng.uniform(0.85, 1.15, n)
    fuel_liter = (
        operating_hours
        * liter_per_hour
        * (1.0 + 0.02 * age)
        * rng.uniform(0.92, 1.08, n)
        * fuel_mult
    )
    maintenance_cost = np.select(
        [is_failure, is_preventive, is_routine],
        [
            rng.uniform(8000, 30000, n),
            rng.uniform(1500, 5000, n),
            rng.uniform(200, 800, n),
        ],
        default=0.0,
    )
    operating_cost = (
        fuel_liter * FUEL_PRICE_USD
        + workers * LABOUR_COST_USD_PER_WORKER
        + maintenance_cost
        + OVERHEAD_USD
    )

    ops_df = pd.DataFrame(
        {
            "date": np.tile(dates, n_eq),
            "site_id": np.array(SITES)[site_idx],
            "equipment_id": roster["equipment_id"].to_numpy()[eq_idx],
            "equipment_type": type_of_eq[eq_idx],
            "age_years": age.astype(int),
            "operating_hours": np.round(operating_hours, 2),
            "production_ton": np.round(production_ton, 1),
            "fuel_liter": np.round(fuel_liter, 1),
            "downtime_hours": np.round(downtime_hours, 2),
            "downtime_reason": downtime_reason,
            "maintenance_type": maintenance_type,
            "maintenance_cost_usd": np.round(maintenance_cost, 2),
            "workers_on_shift": workers,
            "temp_c": np.round(temp, 1),
            "rainfall_mm": np.round(rainfall, 1),
            "ore_grade_pct": np.round(ore_grade, 2),
            "operating_cost_usd": np.round(operating_cost, 2),
            "target_ton": np.round(target_ton, 1),
        }
    )[OPS_COLUMNS]
    # Date-major order for a cleaner CSV; deterministic under a fixed seed.
    ops_df = ops_df.sort_values(["date", "equipment_id"]).reset_index(drop=True)

    # ------------------------------------------------ ore quality fallback table
    # One row per site x day with simple, learnable correlations:
    # higher reagent/air flow, richer iron feed and higher pH all push the
    # silica-in-concentrate target down.
    silica_base = rng.uniform(2.8, 4.2, size=3)
    iron_feed = np.clip(rng.normal(60.0, 2.5, (n_days, 3)), 52, 68)
    reagent = np.clip(rng.normal(750.0, 60.0, (n_days, 3)), 500, 1000)
    air = np.clip(rng.normal(300.0, 25.0, (n_days, 3)), 200, 400)
    ph = np.clip(rng.normal(8.8, 0.35, (n_days, 3)), 7.5, 10.0)
    density = np.clip(rng.normal(1.38, 0.06, (n_days, 3)), 1.2, 1.6)
    silica = np.clip(
        silica_base[None, :]
        - 0.0035 * (reagent - 750.0)
        - 0.0040 * (air - 300.0)
        - 0.05 * (iron_feed - 60.0)
        - 0.15 * (ph - 8.8)
        + rng.normal(0, 0.25, (n_days, 3)),
        0.8,
        6.5,
    )

    ore_df = pd.DataFrame(
        {
            "date": np.tile(dates, 3),
            "site_id": np.repeat(np.array(SITES), n_days),
            "silica_pct": np.round(silica.T.reshape(-1), 2),
            "iron_feed_pct": np.round(iron_feed.T.reshape(-1), 2),
            "reagent_flow": np.round(reagent.T.reshape(-1), 1),
            "air_flow": np.round(air.T.reshape(-1), 1),
            "ph": np.round(ph.T.reshape(-1), 2),
            "density": np.round(density.T.reshape(-1), 3),
        }
    ).sort_values(["date", "site_id"], kind="stable")[ORE_COLUMNS].reset_index(drop=True)

    return ops_df, ore_df


# ---------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic mining-operations datasets."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: <repo>/data)",
    )
    args = parser.parse_args()

    ops_df, ore_df = generate(args.seed)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ops_path = out_dir / "mining_operations.csv"
    ore_path = out_dir / "ore_quality_synthetic.csv"
    ops_df.to_csv(ops_path, index=False)
    ore_df.to_csv(ore_path, index=False)

    fail_rate = (ops_df["downtime_reason"] == "mechanical_failure").mean()
    print(f"seed={args.seed}")
    print(f"wrote {ops_path} ({len(ops_df):,} rows, failure rate {fail_rate:.2%})")
    print(f"wrote {ore_path} ({len(ore_df):,} rows)")


if __name__ == "__main__":
    main()
