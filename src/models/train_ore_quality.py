"""Train the ore-quality prediction model (Task 6, Fase 2).

Predicts the hourly mean ``silica_pct`` of the flotation concentrate from
aggregated process sensors with a ``HistGradientBoostingRegressor``.

Data comes from ``src.data_loader.load_ore_quality``: the real Kaggle export
(``data/raw/flotation.csv``) when present, otherwise the synthetic fallback.
Features are built by ``src.features.make_ore_frame`` (hourly aggregation).

Splits are purely chronological (first 80% of hours -> train, last 20% ->
test; no shuffling). The model is compared against a naive baseline that
predicts the train-set mean, and the bundle (model + feature columns +
target + data source) is saved to ``models/ore_quality.pkl``.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader import load_ore_quality
from src.features import make_ore_frame

TARGET = "silica_pct"
RAW_PATH = os.path.join("data", "raw", "flotation.csv")
FALLBACK_PATH = os.path.join("data", "ore_quality_synthetic.csv")
MODEL_PATH = os.path.join("models", "ore_quality.pkl")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric feature columns: everything except the target and the date."""
    return [
        col
        for col in frame.columns
        if col != TARGET
        and col != "date"
        and pd.api.types.is_numeric_dtype(frame[col])
    ]


def time_split(frame: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: first ``frac`` rows -> train, rest -> test."""
    frame = frame.sort_values("date").reset_index(drop=True)
    cut = int(len(frame) * frac)
    return frame.iloc[:cut], frame.iloc[cut:]


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (RMSE, R^2)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2


def main() -> None:
    ore, source = load_ore_quality(RAW_PATH, FALLBACK_PATH)
    print(f"Ore quality data source: {source} ({len(ore):,} raw rows)")

    frame = make_ore_frame(ore).dropna(subset=[TARGET]).reset_index(drop=True)
    feats = feature_columns(frame)
    if not feats:
        raise RuntimeError("No feature columns available for the ore quality model.")
    print(f"Hourly frame: {frame.shape[0]:,} rows x {len(feats)} features")

    train, test = time_split(frame)
    X_train, y_train = train[feats], train[TARGET]
    X_test, y_test = test[feats], test[TARGET]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    rmse, r2 = metrics(y_test, model.predict(X_test))

    # Baseline: naive prediction of the train-set mean.
    base_rmse, base_r2 = metrics(
        y_test, np.full(len(y_test), float(y_train.mean()))
    )

    print(f"Model            RMSE={rmse:.4f}  R^2={r2:.4f}")
    print(f"Baseline (mean)  RMSE={base_rmse:.4f}  R^2={base_r2:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feats,
            "target": TARGET,
            "source": source,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
