import joblib
import numpy as np
import pandas as pd
from utils.io import load_processed
from utils.preprocess import TARGET_MAIN, TARGET_LOG

ART_PATHS = [
    "models/artifacts/rf_pipeline.pkl",
    "models/artifacts/ridge_pipeline.pkl",
]

def load_best():
    for p in ART_PATHS:
        try:
            return joblib.load(p), p
        except Exception:
            continue
    raise FileNotFoundError("No model artifact found.")

def predict_next(df_last_row: pd.Series, override: dict | None = None) -> dict:
    model, path = load_best()
    x = df_last_row.copy()
    if override:
        for k,v in override.items():
            x[k] = v
    y_pred = model.predict(pd.DataFrame([x]))[0]
    # If the model was trained on log, app should invert; but we don't know here.
    # Provide raw prediction and let UI handle inversion if known.
    return {"artifact": path, "prediction": float(y_pred)}
