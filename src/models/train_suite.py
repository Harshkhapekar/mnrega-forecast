# src/models/train_suite.py
from __future__ import annotations

import os, json, joblib, numpy as np, pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error
from scipy.stats import loguniform, randint
import lightgbm as lgb  # pip install -U lightgbm

from src.utils.io import load_processed
from src.pipelines.targets import build_feature_frame

# Fallback-safe RMSE (silences sklearn deprecation)
try:
    from sklearn.metrics import root_mean_squared_error as rmse_func
    def RMSE(y_true, y_pred): return float(rmse_func(y_true, y_pred))
except Exception:
    from sklearn.metrics import mean_squared_error
    def RMSE(y_true, y_pred): return float(mean_squared_error(y_true, y_pred, squared=False))

TARGETS: Dict[str,str] = {
    "demand": "Total_Households_Worked",
    "utilization": "utilization_target",
    "completion": "completion_target",
}

IGNORE = {"fin_year","month","district_name","state_name","Remarks","completion_ratio"}
BAD_INTERNAL = {"y_hat_seasonal"}

FAST_TRIALS = 6          # randomized search trials (was 20)
CV_STEP_MONTHS = 6       # wider step -> fewer folds (was 3)
BOOST_ROUNDS = 1200      # max boosting rounds (was 5000)
EARLY_STOP = 100         # early stopping rounds (was 200)
FEATURE_CAP = 60         # keep top-K most variable features for speed

@dataclass
class CVResult:
    rmse_list: List[float]
    mae_list: List[float]
    baseline_rmse_list: List[float]
    residuals: List[float]

def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    ds = pd.to_datetime(df["fin_year"].astype(str).str[:4] + "-" + df["month"].astype(int).astype(str) + "-01")
    return df.assign(_ds=ds).sort_values("_ds")

def _seasonal_naive(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.sort_values(["district_name","_ds"]).copy()
    df["y_hat_seasonal"] = df.groupby("district_name")[target].shift(12)
    return df

def _cutoffs(df: pd.DataFrame, warmup: int = 18, step: int = CV_STEP_MONTHS) -> List[pd.Timestamp]:
    uniq = df["_ds"].drop_duplicates().sort_values().to_list()
    if len(uniq) <= warmup + step: return []
    return [uniq[i] for i in range(warmup, len(uniq)-step, step)]

def _feature_list(df: pd.DataFrame, target: str) -> List[str]:
    cols = [c for c in df.columns if c not in IGNORE and c != target and not c.startswith("_") and c not in BAD_INTERNAL]
    # Cap features for speed by variance proxy
    if len(cols) > FEATURE_CAP:
        var = df[cols].var(numeric_only=True).fillna(0.0)
        keep = var.sort_values(ascending=False).index[:FEATURE_CAP].tolist()
        cols = [c for c in cols if c in keep]
    return cols

def _prior_for(key: str) -> Dict:
    base = dict(
        learning_rate=loguniform(0.02, 0.15),
        num_leaves=randint(16, 64),
        max_depth=randint(3, 10),
        min_data_in_leaf=randint(20, 200),
        feature_fraction=loguniform(0.7, 1.0),
        bagging_fraction=loguniform(0.7, 1.0),
        bagging_freq=randint(1, 6),
        lambda_l1=loguniform(1e-3, 5),
        lambda_l2=loguniform(1e-3, 5),
    )
    if key == "utilization":
        base["max_depth"] = randint(3, 7)
        base["num_leaves"] = randint(16, 48)
    return base

def _cv_eval(df: pd.DataFrame, Xcols: List[str], target: str, params: Dict, with_baseline: bool) -> CVResult:
    rmse_, mae_, base_, residuals_ = [], [], [], []
    p = dict(params); p.update({"objective":"regression","metric":"rmse","seed":42,"verbosity":-1})

    for c in _cutoffs(df):
        tr = df["_ds"] <= c
        va = (df["_ds"] > c) & (df["_ds"] <= c + pd.DateOffset(months=3))
        Xtr, ytr = df.loc[tr, Xcols], df.loc[tr, target]
        Xva, yva = df.loc[va, Xcols], df.loc[va, target]
        if Xva.empty: continue

        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr)
        model = lgb.train(
            p,
            train_set=dtr,
            num_boost_round=BOOST_ROUNDS,
            valid_sets=[dva],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred = model.predict(Xva, num_iteration=model.best_iteration)
        rmse_.append(RMSE(yva, pred))
        mae_.append(mean_absolute_error(yva, pred))
        residuals_.extend((yva - pred).tolist())

        if with_baseline:
            base = df.loc[va, "y_hat_seasonal"].fillna(ytr.tail(12).mean())
            base_.append(RMSE(yva, base))

    return CVResult(rmse_, mae_, base_, residuals_)

def _conformal(residuals: List[float]) -> Tuple[float,float]:
    if not residuals: return 0.0, 0.0
    res = np.abs(np.asarray(residuals, dtype=float))
    return float(np.quantile(res, 0.80)), float(np.quantile(res, 0.95))

def train_one(df_feat: pd.DataFrame, key: str, outdir: str):
    target = TARGETS[key]
    df = _date_index(df_feat)
    with_baseline = (key == "demand")
    if with_baseline: df = _seasonal_naive(df, target)
    Xcols = _feature_list(df, target)

    prior = _prior_for(key)
    rng = np.random.RandomState(42)
    candidates = list(ParameterSampler(prior, n_iter=FAST_TRIALS, random_state=rng))

    best_params, best_res, best_score = None, None, float("inf")
    for params in candidates:
        res = _cv_eval(df, Xcols, target, params, with_baseline)
        if not res.rmse_list: continue
        score = float(np.mean(res.rmse_list))
        if score < best_score:
            best_score, best_params, best_res = score, params, res

    if best_params is None:
        best_params = {"learning_rate":0.08, "num_leaves":32, "max_depth":-1, "min_data_in_leaf":50,
                       "feature_fraction":0.9, "bagging_fraction":0.9, "bagging_freq":3, "lambda_l1":0.0, "lambda_l2":0.0}
        best_res = _cv_eval(df, Xcols, target, best_params, with_baseline)

    q80, q95 = _conformal(best_res.residuals if best_res else [])

    final_params = dict(best_params)
    final_params.update({"objective":"regression","metric":"rmse","seed":42,"verbosity":-1})
    dtrain = lgb.Dataset(df[Xcols], label=df[target])
    final = lgb.train(
        final_params,
        train_set=dtrain,
        num_boost_round=BOOST_ROUNDS,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(final, os.path.join(outdir, "model.joblib"))
    json.dump({"features": Xcols, "best_params": best_params}, open(os.path.join(outdir,"feature_spec.json"), "w"), indent=2)
    json.dump({"q80": q80, "q95": q95}, open(os.path.join(outdir,"conformal_q.json"), "w"), indent=2)

    metrics = {
        "rmse_mean": float(np.mean(best_res.rmse_list)) if best_res and best_res.rmse_list else None,
        "rmse_std": float(np.std(best_res.rmse_list)) if best_res and best_res.rmse_list else None,
        "mae_mean": float(np.mean(best_res.mae_list)) if best_res and best_res.mae_list else None,
    }
    if best_res and best_res.baseline_rmse_list:
        metrics["baseline_rmse_mean"] = float(np.mean(best_res.baseline_rmse_list))
    json.dump(metrics, open(os.path.join(outdir,"metrics_cv.json"), "w"), indent=2)

def main():
    raw = load_processed()
    feat = build_feature_frame(raw)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_dir = os.path.join("models","artifacts", ts)
    for key in TARGETS:
        train_one(feat, key, os.path.join(base_dir, key))
    print("Saved suite to", base_dir)

if __name__ == "__main__":
    main()
