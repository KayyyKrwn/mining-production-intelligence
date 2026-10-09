# Data Dictionary

## Generator calibration parameters

All values below are documented so the synthetic data can be audited. They are
calibrated against public industry benchmarks (e.g. haul-truck fuel burn on the
order of 300 L/h; open-pit daily output in the tens of thousands of tonnes per
site). The generator is seeded (`seed=42`); two runs with the same seed produce
bit-identical CSVs.

| Parameter | Value |
|---|---|
| Sites | SITE-A (target 12,000 t/day), SITE-B (8,000), SITE-C (5,000) |
| Fleet | 40 units: haul_truck ×16, excavator ×8, drill ×8, loader ×8; age 1–12 yrs |
| Base operating hours/day | truck 20, excavator 18, drill 14, loader 16 (± noise; reduced in rain) |
| Tonnes per operating hour | truck 320, excavator 450, drill 60, loader 280 (× noise 0.85–1.15) |
| Fuel liters per operating hour | truck 300, excavator 180, drill 90, loader 120 (× age factor 1+0.02·age) |
| Rainy season | Nov–Mar (high rainfall_mm → fewer operating hours, more weather downtime) |
| Failure rate | ~1.5% of equipment-days (`mechanical_failure` + `corrective`); 7–14 days prior show degradation (fuel ×1.15–1.3, 3× minor-issue rate) |
| Maintenance cost (USD) | routine 200–800, preventive 1,500–5,000, corrective 8,000–30,000 |
| Operating cost (USD/row) | fuel×0.9 + workers×25 + maintenance_cost + 500 overhead |
| `target_ton` | site daily target repeated on every equipment row — always aggregate with `.sum()`, never `.mean()` |

## `data/mining_operations.csv` (synthetic)

One row = one equipment × one day.

| Column | Type | Description |
|---|---|---|
| date | date | Operation date (2024-01-01 → 2025-12-31) |
| site_id | string | SITE-A / SITE-B / SITE-C |
| equipment_id | string | Unique equipment code |
| equipment_type | string | haul_truck / excavator / drill / loader |
| age_years | int | Equipment age in years |
| operating_hours | float | Hours operated that day |
| production_ton | float | Tonnes produced that day |
| fuel_liter | float | Fuel consumed (liters) |
| downtime_hours | float | Downtime hours |
| downtime_reason | string | mechanical_minor / mechanical_failure / weather / safety / planned |
| maintenance_type | string | none / routine / preventive / corrective |
| maintenance_cost_usd | float | Maintenance cost (USD) |
| workers_on_shift | int | Workers on shift |
| temp_c | float | Ambient temperature (°C) |
| rainfall_mm | float | Rainfall (mm) |
| ore_grade_pct | float | Ore grade (% Fe) |
| operating_cost_usd | float | Total operating cost (USD) |
| target_ton | float | Daily production target (tonnes) |

## `data/ore_quality_synthetic.csv` (synthetic fallback)

| Column | Type | Description |
|---|---|---|
| date | date | Date |
| site_id | string | Site |
| silica_pct | float | Silica in concentrate (%) — prediction target |
| iron_feed_pct | float | Iron in feed (%) |
| reagent_flow | float | Reagent flow rate |
| air_flow | float | Air flow rate |
| ph | float | Pulp pH |
| density | float | Pulp density |

## `data/raw/flotation.csv` (real, Kaggle — optional)

"Quality Prediction in a Mining Process" (edumagalhaes). 737,453 rows × 24 columns from an iron-ore flotation plant (Mar–Sep 2017). **Note:** the file uses a comma as the decimal separator — load with `decimal=","`.

Target: `% Silica Concentrate`.
