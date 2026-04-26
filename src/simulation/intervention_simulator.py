"""What-if intervention simulation utilities."""

import pandas as pd


def simulate_green_cover_increase(df: pd.DataFrame, increase_percent: float) -> pd.DataFrame:
    """Simulate increased green-cover intervention."""
    simulated = df.copy()
    simulated["green_cover_percent"] = simulated["green_cover_percent"] + increase_percent
    return simulated


def simulate_traffic_reduction(df: pd.DataFrame, reduction_fraction: float) -> pd.DataFrame:
    """Simulate traffic reduction intervention."""
    simulated = df.copy()
    simulated["traffic_index"] = simulated["traffic_index"] * (1 - reduction_fraction)
    return simulated
