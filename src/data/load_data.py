"""Data loading utilities."""

from pathlib import Path
import pandas as pd


def load_sample_data(path: str = "data/sample/sample_aqi_data.csv") -> pd.DataFrame:
    """Load sample AQI data."""
    return pd.read_csv(Path(path))
