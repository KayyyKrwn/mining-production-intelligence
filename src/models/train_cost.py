"""Train the operating-cost regression model (Task 5, Fase 2).

Builds a site-day frame via :func:`src.features.make_cost_frame`, splits it
80/20 in time order (no shuffle), trains a HistGradientBoostingRegressor on
the first 80 % and evaluates it on the last 20 % against a naive baseline
(mean of the target on the training split).

Running this file directly writes ``models/cost.pkl``.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader import load_operations  # noqa: E402
from src.features import make_cost_frame  # noqa: E402

TARGET = "operating_cost_usd"


def repo_root() -> str:
    """Return the repository root directory."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def build_cost_matrix(frame: pd.DataFrame) -> "tuple[pd.DataFrame, pd.Series]":
    """Turn the cost frame into a modelling matrix ``(X, y)``.

    Adds calendar features (day-of-week, month) and one-hot encodes
    ``site_id`` so that site-level cost differences are learnable. The
    result keeps the original row order (time order per site).
    """
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    features = df.drop(columns=[TARGET, "date"])
    features = pd.get_dummies(features, columns=["site_id"], dtype=np.float64)
    return features, df[TARGET]


def time_split(
    X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]":
    """Split in time order (no shuffle): first 80 % train, last 20 % test."""
    cut = int(len(X) * train_frac)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def main() -> None:
    root = repo_root()
    ops = load_operations(os.path.join(root, "data", "mining_operations.csv"))
    frame = make_cost_frame(ops)
    X, y = build_cost_matrix(frame)
    X_train, X_test, y_train, y_test = time_split(X, y)

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    # Naive baseline: constant mean of the training target.
    baseline_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_r2 = float(r2_score(y_test, baseline_pred))

    print(f"[cost] rows: {len(frame)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"[cost] model     -> MAE: {mae:,.2f} USD | R2: {r2:.4f}")
    print(
        f"[cost] baseline   -> MAE: {baseline_mae:,.2f} USD | R2: {baseline_r2:.4f}"
    )
    print(
        f"[cost] improvement: MAE {100.0 * (baseline_mae - mae) / baseline_mae:.1f}% "
        f"lower than baseline"
    )

    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    out_path = os.path.join(root, "models", "cost.pkl")
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "metrics": {
                "mae": mae,
                "r2": r2,
                "baseline_mae": baseline_mae,
                "baseline_r2": baseline_r2,
            },
        },
        out_path,
    )
    print(f"[cost] saved model to {out_path}")


if __name__ == "__main__":
    main()
