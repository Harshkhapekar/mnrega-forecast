import json
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

def evaluate(y_true, y_pred):
    """
    Compute R^2 and RMSE on arrays (original target scale).
    """
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"r2": float(r2), "rmse": float(rmse)}

def residual_std(y_true, y_pred):
    """
    Standard deviation of residuals (y_true - y_pred) on original scale.
    Useful for uncertainty bands.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.std(y_true - y_pred))

def save_metrics(metrics: dict, path: str):
    """
    Save metrics (including optional residual_std and best_model) to JSON.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
