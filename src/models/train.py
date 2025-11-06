import os
import math
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.utils.io import load_raw, save_processed
from src.utils.preprocess import preprocess, TARGET_MAIN, TARGET_LOG
from src.utils.features import build_features
from src.utils.metrics import evaluate, save_metrics  # evaluate uses r2 and RMSE (root_mean_squared_error)

ART_DIR = "models/artifacts"
MET_DIR = "models/metrics"
SCHEMA_PATH = os.path.join(MET_DIR, "schema.json")

def _fy_head4(s: str) -> int:
    s = str(s)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return 0

def make_folds(df: pd.DataFrame, n_splits: int = 5):
    # Use fin_year (as string) + month (int) to build an increasing key
    key = df.apply(lambda r: _fy_head4(r["fin_year"]) * 100 + int(r["month"]), axis=1)
    order = key.sort_values().index
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, te_idx in tss.split(order):
        yield order[tr_idx], order[te_idx]

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # Adds lag1 and rolling mean(3) of the main target per district to inject temporal signal
    if TARGET_MAIN not in df.columns:
        return df
    df = df.copy()
    # Sort by district and time
    df["_fy_key"] = df["fin_year"].apply(_fy_head4)
    df = df.sort_values(["district_name", "_fy_key", "month"])
    # Per-district lags
    grp = df.groupby("district_name", group_keys=False)
    df["lag1_households"] = grp[TARGET_MAIN].shift(1)
    df["roll3_households"] = grp[TARGET_MAIN].rolling(3).mean().reset_index(level=0, drop=True)
    # Clean helper
    df = df.drop(columns=["_fy_key"])
    # Fill remaining NA in engineered lags with group medians to keep rows; alternatively drop NAs for strict CV
    df["lag1_households"] = df.groupby("district_name")["lag1_households"].transform(
        lambda s: s.fillna(s.median())
    )
    df["roll3_households"] = df.groupby("district_name")["roll3_households"].transform(
        lambda s: s.fillna(s.median())
    )
    return df

def build_preprocessor(df: pd.DataFrame):
    drop_cols = [TARGET_MAIN, TARGET_LOG]
    usable = [c for c in df.columns if c not in drop_cols]
    base_cat = ["fin_year", "district_name", "state_name"]

    # Numeric columns: strictly numeric dtype
    num_cols = [c for c in usable if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if c not in base_cat]

    # Categorical columns: everything else (e.g., Remarks)
    cat_cols = [c for c in usable if c not in num_cols]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return pre, num_cols, cat_cols

def save_schema(feature_columns):
    os.makedirs(MET_DIR, exist_ok=True)
    with open(SCHEMA_PATH, "w") as f:
        json.dump({"feature_columns": feature_columns}, f, indent=2)

def main():
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(MET_DIR, exist_ok=True)

    # 1) Load, preprocess, engineer
    df = load_raw()
    df = preprocess(df)          # month/fin_year normalized; numerics coerced
    df = build_features(df)      # ratios, shares, payment_eff, seasonality if any
    df = add_lag_features(df)    # new temporal features

    # Persist processed for the app
    save_processed(df)

    # 2) Target
    use_log = TARGET_LOG in df.columns
    y = df[TARGET_LOG] if use_log else df[TARGET_MAIN]

    # 3) Preprocessor and models
    pre, num_cols, cat_cols = build_preprocessor(df)
    feature_columns = num_cols + cat_cols  # order not critical; names are used by transformers
    save_schema(feature_columns)

    ridge = Pipeline([
        ("pre", pre),
        ("mdl", Ridge(alpha=1.0, random_state=42))
    ])
    rf = Pipeline([
        ("pre", pre),
        ("mdl", RandomForestRegressor(
            n_estimators=700, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
        ))
    ])
    hgb = Pipeline([
        ("pre", pre),
        ("mdl", HistGradientBoostingRegressor(
            learning_rate=0.08, max_depth=None, max_iter=500, l2_regularization=0.0,
            early_stopping=True, random_state=42
        ))
    ])

    candidates = [("ridge", ridge), ("rf", rf), ("hgb", hgb)]

    # 4) Time-aware CV, collect residuals for intervals
    metrics_all = {}
    best_model = None
    best_score = -1e9

    # Accumulate out-of-fold residuals on original scale
    oof_true = []
    oof_pred = []

    for name, pipe in candidates:
        preds = pd.Series(index=df.index, dtype=float)
        for tr, te in make_folds(df):
            pipe.fit(df.loc[tr, :], y.loc[tr])
            fold_pred = pipe.predict(df.loc[te, :])
            preds.loc[te] = fold_pred

        if use_log:
            y_true = df[TARGET_MAIN].values
            y_pred = preds.fillna(0.0).clip(-5, 20).apply(lambda z: math.exp(float(z))).values
        else:
            y_true = y.values
            y_pred = preds.values

        m = evaluate(y_true, y_pred)
        metrics_all[name] = m

        # Track OOF residuals for the best-so-far model
        if m["r2"] > best_score:
            best_score = m["r2"]
            best_model = (name, pipe)
            oof_true = list(y_true)
            oof_pred = list(y_pred)

    # 5) Fit best on full data and persist
    name, pipe = best_model
    pipe.fit(df, y)
    joblib.dump(pipe, os.path.join(ART_DIR, f"{name}_pipeline.pkl"))

    # Compute residual std for intervals from best model's OOF preds
    import numpy as np
    residual_std = float(np.std(np.array(oof_true) - np.array(oof_pred)))

    # Save metrics plus residual_std
    metrics_all["best_model"] = name
    metrics_all["residual_std"] = residual_std
    save_metrics(metrics_all, os.path.join(MET_DIR, "cv_metrics.json"))
    print("Saved best model:", name, "Metrics:", metrics_all)

if __name__ == "__main__":
    main()
