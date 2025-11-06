import os
import pandas as pd

RAW_PATH = "data/raw/mnrega_clean.csv"
PROCESSED_PATH = "data/processed/processed.parquet"

def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)

def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def load_processed(path: str = PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
