"""Optimization logic for choosing intervention strategies."""


def choose_best_strategy(strategy_results):
    """Choose strategy with lowest predicted AQI."""
    return min(strategy_results, key=lambda item: item["predicted_aqi"])
