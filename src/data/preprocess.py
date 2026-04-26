"""Data preprocessing utilities."""

import pandas as pd


def preprocess_aqi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing for AQI data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    return df
