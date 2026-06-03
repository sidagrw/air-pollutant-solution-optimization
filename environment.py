# environment.py
import numpy as np
from config import EnvironmentConfig


class SurrogateEnv:
    def __init__(self, config: EnvironmentConfig):
        self.dt = config.dt
        self.decay_rate = config.decay_rate
        self.scrubber_efficiency = config.scrubber_efficiency
        self.pm10_limit = config.pm10_limit
        self.pm10_floor = config.pm10_floor
        self.initial_pm10 = config.initial_pm10
        self.current_pm10 = self.initial_pm10
        self.timestep_counter = 0

    def reset(self, initial_pm10: float = None) -> float:
        if initial_pm10 is not None:
            self.current_pm10 = max(self.pm10_floor, initial_pm10)
        else:
            self.current_pm10 = self.initial_pm10
        self.timestep_counter = 0
        return self.current_pm10

    def _calculate_next_state(self, scrubber_action: float, ambient_pm10_inflow: float) -> float:
        inflow = ambient_pm10_inflow
        natural_settling = self.decay_rate * self.current_pm10
        scrubber_removal = self.scrubber_efficiency * scrubber_action
        net_change = (inflow - natural_settling - scrubber_removal) * self.dt
        return max(self.pm10_floor, self.current_pm10 + net_change)

    def _compute_step_penalties(self, scrubber_action: float) -> dict:
        return {
            "safety_violation": max(0.0, self.current_pm10 - self.pm10_limit),
            "water_pumping_cost": float(scrubber_action ** 2)
        }

    def step(self, scrubber_action: float, ambient_pm10_inflow: float) -> dict:
        self.current_pm10 = self._calculate_next_state(scrubber_action, ambient_pm10_inflow)
        penalties = self._compute_step_penalties(scrubber_action)
        self.timestep_counter += 1
        return {
            "next_state": self.current_pm10,
            "penalties": penalties,
            "timestep": self.timestep_counter
        }
