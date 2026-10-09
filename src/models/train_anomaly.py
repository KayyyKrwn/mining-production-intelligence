"""Train an unsupervised anomaly detector on equipment-day operating data.

Phase 2 - Model 3 (anomaly detection). Fits an IsolationForest with
``contamination=0.02`` on the equipment-day frame produced by
:func:`src.features.make_anomaly_frame` and saves it to
``models/anomaly.pkl``.

Evaluation is precision@k against known mechanical failures: among the
top-2% most anomalous equipment-days (lowest ``decision_function`` scores),
what fraction coincide with a row labelled ``mechanical_failure`` in
``downtime_reason``.

Usage:
    python src/models/train_anomaly.py   (run from the repo root)
"""

import os
import sys

import joblib
from sklearn.ensemble import IsolationForest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data_loader import load_operations
from src.features import make_anomaly_frame

CONTAMINATION = 0.02
FEATURES = ["fuel_per_ton", "cost_per_ton", "downtime_hours", "operating_hours"]
MODEL_PATH = os.path.join("models", "anomaly.pkl")


def main() -> dict:
    """Train, evaluate (precision@k) and save the anomaly model."""
    ops = load_operations(os.path.join("data", "mining_operations.csv"))
    frame = make_anomaly_frame(ops)

    X = frame[FEATURES].to_numpy(dtype=float)
    model = IsolationForest(contamination=CONTAMINATION, random_state=42)
    model.fit(X)

    # --- precision@k: align failure labels via (equipment_id, date) keys ---
    labels = (
        ops[["equipment_id", "date"]]
        .assign(is_failure=(ops["downtime_reason"] == "mechanical_failure").astype(int))
        .groupby(["equipment_id", "date"], as_index=False)["is_failure"]
        .max()
    )
    scored = frame.merge(labels, on=["equipment_id", "date"], how="left").fillna(
        {"is_failure": 0}
    )
    scored["anomaly_score"] = model.decision_function(X)

    n = len(scored)
    k = max(1, int(CONTAMINATION * n))
    top_k = scored.nsmallest(k, "anomaly_score")
    precision_at_k = float(top_k["is_failure"].mean())
    prevalence = float(scored["is_failure"].mean())

    payload = {
        "model": model,
        "features": FEATURES,
        "contamination": CONTAMINATION,
        "trained_rows": n,
        "precision_at_k": precision_at_k,
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    print(f"Trained IsolationForest (contamination={CONTAMINATION}) on {n:,} rows.")
    print(
        f"precision@k (top {CONTAMINATION:.0%} anomalies, k={k:,}): "
        f"{precision_at_k:.3f} "
        f"({int(top_k['is_failure'].sum())} of {k:,} flagged days had a mechanical failure)"
    )
    print(f"Random baseline (failure prevalence): {prevalence:.3f}")
    print(f"Saved: {MODEL_PATH}")
    return payload


if __name__ == "__main__":
    main()
