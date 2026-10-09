# Model Metrics — Fase 2 (vs Baseline)

All supervised models use a **chronological 80/20 split with no shuffling**
(first 80% of time → train, last 20% → test). Numbers below are from the
verified training run via `scripts/train_all.py` (sklearn, `random_state=42`,
deterministic).

| Model | Metric (temporal holdout) | Baseline | Model | Data source |
|---|---|---|---|---|
| Production forecasting | MAE / RMSE (ton) | Naive lag-7: 3,413.01 / 4,581.50 | 2,156.86 / 2,935.20 | Synthetic ops (site-day) |
| Equipment failure prediction | PR-AUC / ROC-AUC | Dummy stratified: 0.1138 / 0.4935 | 0.6764 / 0.9393 | Synthetic ops (simulated labels) |
| Anomaly detection | precision@k top-2% (in-sample) | Random: 0.015 | 0.750 | Synthetic ops |
| Cost prediction | MAE (USD) / R² | Mean-train: 39,962.52 / −38.52 | 588.77 / 0.9729 | Synthetic ops (site-day) |
| Ore quality prediction | RMSE / R² (hourly) | Mean-train: 1.1447 / −0.0013 | 0.7170 / 0.6072 | Kaggle flotation (737k rows → 4,097 hourly) |

## Notes

- **Evaluation protocol:** every supervised model is evaluated on a temporal
  holdout (last 20% of the time range) and compared against a naive baseline
  (naive lag-7 / stratified dummy / constant train mean). PR-AUC is used for
  failure prediction because positives are rare (~1.5%).
- **Failure labels are simulated labels from the generator** (documented
  transparently): the ops generator injects mechanical-failure events with a
  realistic degradation signature (rising fuel/ton, minor downtime,
  time-since-maintenance). The failure model therefore learns to detect the
  generator's own failure signature — useful as a portfolio demonstration,
  not as a production-grade failure predictor.
- **Ore quality** uses the real Kaggle dataset *Quality Prediction in a
  Mining Process* (737,453 rows) aggregated to hourly means (4,097 hours);
  the pipeline falls back to synthetic ore data if the CSV is absent.
- Model artifacts (`models/*.pkl`) are gitignored; retrain with
  `python scripts/train_all.py` from the repo root.
