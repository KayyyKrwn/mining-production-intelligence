"""Single entry point for loading trained models.

Pages should use :func:`load_model` and handle ``FileNotFoundError``
gracefully (show an info message + fall back to the Phase 1 view).
"""

import os

import joblib

MODEL_FILES = {
    "forecast": os.path.join("models", "forecast.pkl"),
    "failure": os.path.join("models", "failure.pkl"),
    "anomaly": os.path.join("models", "anomaly.pkl"),
    "cost": os.path.join("models", "cost.pkl"),
    "ore_quality": os.path.join("models", "ore_quality.pkl"),
}


def _repo_root() -> str:
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def load_model(name: str):
    """Load a trained model by name.

    Raises:
        KeyError: if ``name`` is not a known model.
        FileNotFoundError: if the model file does not exist yet, with a
            friendly message telling the user how to train it.
    """
    if name not in MODEL_FILES:
        raise KeyError(
            f"Unknown model: {name!r}. Available: {sorted(MODEL_FILES)}"
        )
    path = os.path.join(_repo_root(), MODEL_FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Train it first with: python scripts/train_all.py"
        )
    return joblib.load(path)
