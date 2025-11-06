import numpy as np
import pandas as pd

ID_COLS = ["fin_year","month","state_name","district_name"]
TARGET_MAIN = "Total_Households_Worked"
TARGET_LOG = "log_Total_Households_Worked"

# Map month strings to integers 1..12 (handles full/abbr/case-insensitive)
_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _parse_month(m):
    # Already numeric
    if pd.api.types.is_number(m):
        try:
            mi = int(m)
            if 1 <= mi <= 12:
                return mi
        except Exception:
            pass
    # String formats
    s = str(m).strip().lower()
    if s.isdigit():
        mi = int(s)
        return mi if 1 <= mi <= 12 else np.nan
    return _MONTH_MAP.get(s, np.nan)

def _parse_fin_year(fy):
    # Accepts "2019-20", "2019-2020", "2019", 2019
    s = str(fy).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 4:
        return digits[:4]  # keep as string to treat as categorical
    return s or "Unknown"

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure required columns exist
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Normalize IDs
    df["fin_year"] = df["fin_year"].apply(_parse_fin_year).astype(str)
    df["month"] = df["month"].apply(_parse_month)
    # Fill invalid months with mode; if still NaN, set to 1
    if df["month"].isna().any():
        mode_month = df["month"].mode(dropna=True)
        fallback = int(mode_month.iloc[0]) if not mode_month.empty else 1
        df["month"] = df["month"].fillna(fallback).astype(int)

    df["state_name"] = df["state_name"].astype(str)
    df["district_name"] = df["district_name"].astype(str)

    # Convert all other columns (except Remarks) to numeric where possible
    num_cols = [c for c in df.columns if c not in ID_COLS + ["Remarks"]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Median impute numerics
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_types(df)
    df = add_time_features(df)
    return df
