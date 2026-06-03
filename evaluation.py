# evaluation.py

import numpy as np

class Evaluator:

    def __init__(self):

        self.pm10_history = []
        self.action_history = []
        self.violation_history = []

    def record(self, pm10, action, penalties):

        self.pm10_history.append(pm10)
        self.action_history.append(action)
        self.violation_history.append(
            penalties["safety_violation"]
        )

    def compute_average_pm10(self):

        return np.mean(self.pm10_history)

    def compute_total_water_usage(self):

        return np.sum(np.square(self.action_history))

    def compute_total_violations(self):

        return np.sum(self.violation_history)

    def compute_rmse(self, target=50):

        errors = np.array(self.pm10_history) - target

        return np.sqrt(np.mean(errors ** 2))

    def summary(self):

        return {
            "Average PM10": self.compute_average_pm10(),
            "Water Usage": self.compute_total_water_usage(),
            "Total Violations": self.compute_total_violations(),
            "RMSE": self.compute_rmse()
        }