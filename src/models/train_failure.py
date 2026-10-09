"""Train the equipment failure-prediction model.

Target: ``failure_next_7d`` from :func:`src.features.make_failure_frame`
(1 when a mechanical failure occurs on any of the next 7 days, per unit).

Protocol:
- temporal 80/20 split at the *date* level (first 80% of dates train,
  last 20% test; no shuffling, no leakage),
- model: ``HistGradientBoostingClassifier``,
- baseline: ``DummyClassifier(strategy="stratified")`` (PR-AUC ~= prevalence),
- primary metric: PR-AUC (average precision); ROC-AUC reported too.

Run from the repo root::

    python src/models/train_failure.py

Saves a bundle dict to ``models/failure.pkl`` with keys:
``model``, ``feature_cols``, ``label``, ``horizon`` and ``metrics``.
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loader import load_operations
from src.features import make_failure_frame

FEATURE_COLS = [
    "fuel_per_ton",
    "fuel_trend_14d",
    "minor_downtime_14d",
    "hours_since_maintenance",
    "age_years",
    "operating_hours",
]
LABEL = "failure_next_7d"
HORIZON = 7
TRAIN_FRAC = 0.80
MODEL_PATH = os.path.join("models", "failure.pkl")


def temporal_split(frame: pd.DataFrame, frac: float = TRAIN_FRAC):
    """Split equipment-day rows 80/20 at the date level (train = older)."""
    dates = sorted(pd.to_datetime(frame["date"]).unique())
    cutoff = dates[int(len(dates) * frac) - 1]
    dates_s = pd.to_datetime(frame["date"])
    train = frame[dates_s <= cutoff].reset_index(drop=True)
    test = frame[dates_s > cutoff].reset_index(drop=True)
    return train, test, pd.Timestamp(cutoff)


def evaluate(y_true, proba):
    """Return (PR-AUC, ROC-AUC) for predicted probabilities."""
    return (
        float(average_precision_score(y_true, proba)),
        float(roc_auc_score(y_true, proba)),
    )


def main() -> dict:
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(repo_root)

    ops = load_operations(os.path.join("data", "mining_operations.csv"))
    frame = make_failure_frame(ops, horizon=HORIZON)
    print(f"Failure frame: {len(frame):,} equipment-day rows, "
          f"positive rate = {frame[LABEL].mean():.4f}")

    train, test, cutoff = temporal_split(frame)
    print(f"Temporal split at {cutoff.date()}: "
          f"train={len(train):,} rows, test={len(test):,} rows")

    X_train, y_train = train[FEATURE_COLS], train[LABEL].astype(int)
    X_test, y_test = test[FEATURE_COLS], test[LABEL].astype(int)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    pr_model, roc_model = evaluate(y_test, model.predict_proba(X_test)[:, 1])

    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(X_train, y_train)
    pr_base, roc_base = evaluate(y_test, baseline.predict_proba(X_test)[:, 1])

    print(f"Model    : PR-AUC={pr_model:.4f}  ROC-AUC={roc_model:.4f}")
    print(f"Baseline : PR-AUC={pr_base:.4f}  ROC-AUC={roc_base:.4f} "
          f"(stratified dummy)")
    if pr_model <= pr_base:
        print("WARNING: model PR-AUC did not beat the baseline.")

    metrics = {
        "pr_auc_model": pr_model,
        "roc_auc_model": roc_model,
        "pr_auc_baseline": pr_base,
        "roc_auc_baseline": roc_base,
        "train_rows": len(train),
        "test_rows": len(test),
        "cutoff": str(cutoff.date()),
        "positive_rate": float(frame[LABEL].mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "label": LABEL,
        "horizon": HORIZON,
        "metrics": metrics,
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    main()
