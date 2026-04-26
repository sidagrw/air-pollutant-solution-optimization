"""Feature engineering for AQI prediction."""

import pandas as pd


FEATURE_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "temperature",
    "humidity",
    "traffic_index",
    "green_cover_percent",
]


def build_features(df: pd.DataFrame):
    """Return X and y for AQI modeling."""
    X = df[FEATURE_COLUMNS]
    y = df["aqi"]
    return X, y
