# controllers.py

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any


class BaseController(ABC):
    def __init__(self, action_bounds: Tuple[float, float] = (0.0, 100.0)):
        self.action_min, self.action_max = action_bounds
        self.previous_action = 0.0

    @abstractmethod
    def reset(self) -> None:
        self.previous_action = 0.0

    @abstractmethod
    def choose_action(self, current_pm10: float, ambient_inflow: float) -> float:
        pass

    def _clip_action(self, action: float) -> float:
        clipped = float(np.clip(action, self.action_min, self.action_max))
        self.previous_action = clipped
        return clipped


class ThresholdController(BaseController):
    def __init__(self, hysteresis_high: float = 75.0, hysteresis_low: float = 40.0,
                 max_power: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.hysteresis_high = hysteresis_high
        self.hysteresis_low  = hysteresis_low
        self.max_power       = max_power

    def reset(self) -> None:
        super().reset()

    def choose_action(self, current_pm10: float, ambient_inflow: float) -> float:
        if current_pm10 > self.hysteresis_high:
            action = self.max_power
        elif current_pm10 < self.hysteresis_low:
            action = 0.0
        else:
            action = self.previous_action  # hold last action in the dead band
        return self._clip_action(action)


class PIDController(BaseController):
    def __init__(self, kp: float = 2.0, ki: float = 0.1, kd: float = 0.5,
                 target_pm10: float = 50.0, integral_limit: float = 50.0,
                 alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.kp             = kp
        self.ki             = ki
        self.kd             = kd
        self.target_pm10    = target_pm10
        self.integral_limit = integral_limit
        self.alpha          = alpha
        self.integral            = 0.0
        self.previous_error      = 0.0
        self.filtered_derivative = 0.0

    def reset(self) -> None:
        super().reset()
        self.integral            = 0.0
        self.previous_error      = 0.0
        self.filtered_derivative = 0.0

    def choose_action(self, current_pm10: float, ambient_inflow: float) -> float:
        error  = current_pm10 - self.target_pm10
        p_term = self.kp * error

        self.integral = float(np.clip(self.integral + error, -self.integral_limit, self.integral_limit))
        i_term = self.ki * self.integral
        raw_d = error - self.previous_error
        self.filtered_derivative = self.alpha * raw_d + (1 - self.alpha) * self.filtered_derivative
        d_term = self.kd * self.filtered_derivative

        self.previous_error = error
        return self._clip_action(p_term + i_term + d_term)


class MPCController(BaseController):
    def __init__(self, forecaster: Any, horizon: int = 7, target_pm10: float = 50.0,
                 pm10_limit: float = 100.0, action_steps: int = 11, **kwargs):
        super().__init__(**kwargs)
        self.forecaster        = forecaster
        self.horizon           = horizon
        self.target_pm10       = target_pm10
        self.pm10_limit        = pm10_limit
        self.candidate_actions = np.linspace(self.action_min, self.action_max, action_steps)

        self.w_tracking   = 1.0
        self.w_water      = 0.28
        self.w_smoothness = 3.5
        self.w_violation  = 250.0

    def reset(self) -> None:
        super().reset()

    def choose_action(self, current_pm10: float, ambient_inflow: float) -> float:
        n_cand  = len(self.candidate_actions)
        pm_states   = np.full(n_cand, current_pm10)
        total_costs = np.zeros(n_cand)

        total_costs += self.w_smoothness * (self.candidate_actions - self.previous_action) ** 2

        for step in range(self.horizon):
            # once below target, cut the pump to avoid overshoot
            if step > 0:
                actions_used = np.where(pm_states < self.target_pm10, 0.0, self.candidate_actions)
            else:
                actions_used = self.candidate_actions

            X = np.column_stack([pm_states, np.full(n_cand, ambient_inflow), actions_used])
            pm_states = np.maximum(15.0, self.forecaster.model.predict(X))

            total_costs += (
                self.w_tracking  * (pm_states - self.target_pm10) ** 2
                + self.w_water   * actions_used ** 2
                + self.w_violation * np.maximum(0.0, pm_states - self.pm10_limit)
            )

        best_action = self.candidate_actions[np.argmin(total_costs)]
        return self._clip_action(best_action)


class PPOController(BaseController):
    def __init__(self, model_path: str = "models/ppo_aqi_agent.zip",
                 target_pm10: float = 50.0, pm10_scale: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.model_path  = model_path
        self.target_pm10 = target_pm10
        self.pm10_scale  = pm10_scale
        self.model       = self._safe_load_model()

    def _safe_load_model(self) -> Optional[Any]:
        if os.path.exists(self.model_path):
            try:
                from stable_baselines3 import PPO
                return PPO.load(self.model_path)
            except Exception:
                return None
        return None

    def reset(self) -> None:
        super().reset()

    def choose_action(self, current_pm10: float, ambient_inflow: float) -> float:
        if self.model is not None:
            obs = np.array([current_pm10, ambient_inflow, current_pm10/self.pm10_scale], dtype=np.float32)
            action_output, _ = self.model.predict(obs, deterministic=True)
            action = float(action_output) if np.isscalar(action_output) else float(action_output[0])
        else:
            # fallback heuristic when no model is loaded
            error = current_pm10 - self.target_pm10
            if error > 20.0:
                action = self.previous_action + 5.0
            elif error > 5.0:
                action = self.previous_action + 2.0
            elif error < -10.0:
                action = self.previous_action - 5.0
            else:
                action = self.previous_action * 0.94
        return self._clip_action(action)