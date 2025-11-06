import numpy as np
import pandas as pd

def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b==0, np.nan, a / b)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Budget utilization and works ratios
    df["budget_utilization"] = safe_div(df["Wages"], df["Approved_Labour_Budget"]).clip(0, 2)
    total_works = df["Number_of_Completed_Works"] + df["Number_of_Ongoing_Works"]
    df["completed_ratio"] = safe_div(df["Number_of_Completed_Works"], total_works + 1)

    # Social composition shares
    sc = df["SC_persondays"].clip(lower=0)
    st = df["ST_persondays"].clip(lower=0)
    women = df["Women_Persondays"].clip(lower=0)
    denom = sc + st + women + 1
    df["sc_share"] = safe_div(sc, denom)
    df["st_share"] = safe_div(st, denom)
    df["women_share"] = safe_div(women, denom)

    # Payment efficiency
    if "percentage_payments_gererated_within_15_days" in df.columns:
        df["payment_eff"] = df["percentage_payments_gererated_within_15_days"].clip(0, 100)
    else:
        df["payment_eff"] = 0.0

    # Keep target columns if present
    return df
