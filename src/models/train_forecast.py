"""Train the production-forecasting model (Task 2, Fase 2).

Target: site-day ``production_ton`` from ``src.features.make_forecast_frame``.
Split: chronological 80/20 (no shuffling — time series).
Model: ``HistGradientBoostingRegressor`` vs a naive baseline that repeats
the value from 7 days ago. The trained bundle is saved to ``models/forecast.pkl``
and loaded by ``pages/07_Forecast.py`` via ``src.models.inference.load_model``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_operations
from src.features import make_forecast_frame

TARGET = "production_ton"

FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "roll_mean_7",
    "roll_mean_30",
    "dow",
    "month",
    "temp_c",
    "rainfall_mm",
    "n_units",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "mining_operations.csv"
MODEL_PATH = REPO_ROOT / "models" / "forecast.pkl"


def build_frame(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw operations data and build the site-day forecasting frame."""
    ops = load_operations(str(data_path))
    return make_forecast_frame(ops)


def time_split(frame: pd.DataFrame, train_frac: float = 0.8):
    """Chronological 80/20 split (train = first 80% of time)."""
    cutoff = int(len(frame) * train_frac)
    return frame.iloc[:cutoff].copy(), frame.iloc[cutoff:].copy()


def train(frame: pd.DataFrame, random_state: int = 42) -> HistGradientBoostingRegressor:
    """Fit a gradient-boosting regressor on the forecast frame."""
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        early_stopping=True,
        random_state=random_state,
    )
    model.fit(frame[FEATURE_COLS], frame[TARGET])
    return model


def evaluate(model, test: pd.DataFrame) -> dict:
    """Score the model on the holdout set against the naive 7-day-ago baseline."""
    X = test[FEATURE_COLS]
    y = test[TARGET].to_numpy(dtype=float)
    pred = model.predict(X)
    baseline = test["lag_7"].to_numpy(dtype=float)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "baseline_mae": float(mean_absolute_error(y, baseline)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y, baseline))),
    }


def main() -> dict:
    """Train, evaluate vs baseline, save the model bundle, and print metrics."""
    frame = build_frame()
    train_df, test_df = time_split(frame)
    model = train(train_df)
    metrics = evaluate(model, test_df)

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "target": TARGET,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(f"Rows: train={len(train_df)}  holdout={len(test_df)}")
    print(
        f"Model          MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}"
    )
    print(
        f"Naive (7d ago) MAE={metrics['baseline_mae']:.2f}  "
        f"RMSE={metrics['baseline_rmse']:.2f}"
    )
    print(f"Saved: {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    main()
