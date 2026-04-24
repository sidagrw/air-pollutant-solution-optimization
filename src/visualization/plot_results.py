"""Plotting utilities."""

import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_true, y_pred, output_path=None):
    """Plot actual vs predicted AQI."""
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    return plt.gcf()
