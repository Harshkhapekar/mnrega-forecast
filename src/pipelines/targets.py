# src/pipelines/targets.py
import numpy as np, pandas as pd

KEY_DRIVERS = [
    "Approved_Labour_Budget",
    "Average_Wage_rate_per_day_per_person",
    "Total_No_of_Active_Job_Cards",
]

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"Wages","Approved_Labour_Budget"}.issubset(df.columns):
        util = df["Wages"] / df["Approved_Labour_Budget"].replace(0, np.nan)
        df["utilization_target"] = util.clip(0, 1.5).fillna(0.0)
    if "Number_of_Completed_Works" in df:
        df["completion_target"] = df["Number_of_Completed_Works"].astype(float)
    if {"Number_of_Completed_Works","Number_of_Ongoing_Works"}.issubset(df.columns):
        denom = (df["Number_of_Completed_Works"] + df["Number_of_Ongoing_Works"]).replace(0, np.nan)
        df["completion_ratio"] = (df["Number_of_Completed_Works"] / denom).fillna(0.0)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "month" in df:
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
        df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
    return df

def _by(df, group="district_name"):
    return df.sort_values([group, "fin_year", "month"]).groupby(group, group_keys=False)

def add_lags_rolls(df: pd.DataFrame, group="district_name",
                   target="Total_Households_Worked",
                   lags=(1,3,6,12), rolls=(3,6,12)) -> pd.DataFrame:
    df = df.copy(); g = _by(df, group)
    for L in lags:
        df[f"lag{L}_{target}"] = g[target].shift(L)
    for W in rolls:
        df[f"roll{W}_mean_{target}"] = g[target].shift(1).rolling(W).mean()
        df[f"roll{W}_std_{target}"]  = g[target].shift(1).rolling(W).std()
    for col in KEY_DRIVERS + ["Number_of_Completed_Works","Number_of_Ongoing_Works"]:
        if col in df.columns:
            for L in lags:
                df[f"lag{L}_{col}"] = g[col].shift(L)
    return df

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_targets(df)
    df = add_time_features(df)
    df = add_lags_rolls(df)
    df = df.copy(); df["_order"] = df.groupby("district_name").cumcount()
    return df[df["_order"] >= 12].drop(columns=["_order"])
