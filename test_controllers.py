# test_controllers.py
import os
import math
import pytest
import numpy as np

# Import framework components
from config import EnvironmentConfig
from environment import SurrogateEnv
from data_generator import DataGenerator
from evaluation import Evaluator

from controllers import (
    ThresholdController,
    PIDController,
    MPCController,
    PPOController
)

# =====================================================================
# MOCKS & FIXTURES FOR DETERMINISTIC TESTING
# =====================================================================

class DummyForecaster:
    """
    A deterministic, linear surrogate forecaster for testing the MPC 
    without requiring the actual Random Forest .pkl model file.
    Ensures CI/CD pipelines don't fail if the model file is missing.
    """
    def predict(self, current_pm10: float, ambient_inflow: float, scrubber_action: float) -> float:
        # Simple mass-balance surrogate: next = current + inflow - decay - scrubbing
        next_pm10 = current_pm10 + ambient_inflow - (0.03 * current_pm10) - (0.15 * scrubber_action)
        return max(0.0, float(next_pm10))

@pytest.fixture
def dummy_forecaster():
    return DummyForecaster()

@pytest.fixture
def env_config():
    return EnvironmentConfig(initial_pm10=45.0, pm10_limit=100.0)

# =====================================================================
# 1. THRESHOLD CONTROLLER TESTS
# =====================================================================

def test_threshold_hysteresis_logic():
    """Validates hysteresis band memory retention and bang-bang bounds."""
    ctrl = ThresholdController(hysteresis_high=75.0, hysteresis_low=40.0, max_power=100.0)
    
    # 1. Exceed high threshold -> Trigger max power
    action = ctrl.choose_action(current_pm10=80.0, ambient_inflow=10.0)
    assert action == 100.0, "Failed to trigger max power above hysteresis_high."
    
    # 2. Drop into hysteresis band -> Retain previous state (100.0)
    action = ctrl.choose_action(current_pm10=60.0, ambient_inflow=10.0)
    assert action == 100.0, "Failed to retain ON state within hysteresis band."
    
    # 3. Drop below low threshold -> Shut off
    action = ctrl.choose_action(current_pm10=30.0, ambient_inflow=10.0)
    assert action == 0.0, "Failed to shut off below hysteresis_low."
    
    # 4. Rise into hysteresis band -> Retain previous state (0.0)
    action = ctrl.choose_action(current_pm10=60.0, ambient_inflow=10.0)
    assert action == 0.0, "Failed to retain OFF state within hysteresis band."

# =====================================================================
# 2. PID CONTROLLER TESTS
# =====================================================================

def test_pid_stabilization_and_anti_windup():
    """Validates proportional directionality and integral clamping."""
    ctrl = PIDController(kp=1.0, ki=0.5, kd=0.1, target_pm10=50.0, integral_limit=20.0)
    
    # Positive error (PM10 too high) -> Control action should be positive
    action1 = ctrl.choose_action(current_pm10=60.0, ambient_inflow=10.0)
    assert action1 > 0.0, "PID failed to react to positive tracking error."
    
    # Persistently high PM10 -> Action should increase due to integral term
    action2 = ctrl.choose_action(current_pm10=60.0, ambient_inflow=10.0)
    assert action2 > action1, "PID integral term failed to accumulate."
    
    # Test Anti-Windup Limit
    for _ in range(100): 
        ctrl.choose_action(current_pm10=100.0, ambient_inflow=10.0)
    
    assert ctrl.integral <= 20.0, "PID failed to clamp integral windup."
    
    # Negative error (PM10 below target) -> Action should drop towards 0
    action_low = ctrl.choose_action(current_pm10=30.0, ambient_inflow=10.0)
    assert action_low < action2, "PID failed to decelerate on negative error."

# =====================================================================
# 3. MPC CONTROLLER TESTS
# =====================================================================

def test_mpc_determinism_and_bounds(dummy_forecaster):
    """Validates predictive horizon determinism and physical bounds."""
    ctrl = MPCController(forecaster=dummy_forecaster, horizon=3, target_pm10=50.0)
    
    # Reset controller to clear previous_action state, then test
    ctrl.reset()
    action1 = ctrl.choose_action(current_pm10=80.0, ambient_inflow=15.0)
    
    # Reset controller again to ensure identical starting conditions
    ctrl.reset()
    action2 = ctrl.choose_action(current_pm10=80.0, ambient_inflow=15.0)
    
    assert action1 == action2, "MPC is non-deterministic given identical state."
    assert 0.0 <= action1 <= 100.0, "MPC generated an out-of-bounds action."

# =====================================================================
# 4. PPO CONTROLLER TESTS
# =====================================================================

def test_ppo_fallback_operation():
    """Validates safe execution of the adaptive heuristic when model is missing."""
    # Point to a deliberately fake path to trigger fallback
    ctrl = PPOController(model_path="fake_path_does_not_exist.zip", target_pm10=50.0)
    
    assert ctrl.model is None, "PPO Controller loaded a model it shouldn't have."
    
    # Test adaptive fallback step logic (Updated for sluggish calibration)
    ctrl.reset()
    act_high = ctrl.choose_action(current_pm10=80.0, ambient_inflow=15.0) # Error > 20
    assert act_high == 8.0 # Previous (0) + 8
    
    act_med = ctrl.choose_action(current_pm10=60.0, ambient_inflow=15.0)  # Error > 5
    assert act_med == 10.0 # Previous (8) + 2
    
    act_low = ctrl.choose_action(current_pm10=20.0, ambient_inflow=15.0)  # Error < -10
    assert act_low == 5.0 # Previous (10) - 5

# =====================================================================
# 5. INTEGRATION & SAFETY TESTS
# =====================================================================

@pytest.mark.parametrize("controller_class", [
    ThresholdController,
    PIDController,
    MPCController,
    PPOController
])
def test_integration_safety_constraints(controller_class, dummy_forecaster, env_config):
    """
    Runs a simulation loop to ensure physical constraints are NEVER violated:
    - No NaNs
    - Actions bounded [0, 100]
    - PM10 strictly non-negative
    """
    env = SurrogateEnv(env_config)
    generator = DataGenerator(seed=42)
    
    # Initialize controller dynamically
    if controller_class == MPCController:
        ctrl = controller_class(forecaster=dummy_forecaster)
    elif controller_class == PPOController:
        ctrl = controller_class(model_path="fake.zip")
    else:
        ctrl = controller_class()
        
    state = env.reset()
    ctrl.reset()
    
    for t in range(30):
        inflow = generator.traffic_spike(t)
        action = ctrl.choose_action(current_pm10=state, ambient_inflow=inflow)
        
        # 1. Action Safety Bounds
        assert 0.0 <= action <= 100.0, f"[{controller_class.__name__}] Action {action} out of bounds."
        assert not math.isnan(action), f"[{controller_class.__name__}] Action is NaN."
        
        result = env.step(scrubber_action=action, ambient_pm10_inflow=inflow)
        state = result["next_state"]
        
        # 2. Physics Constraints
        assert state >= 0.0, f"[{controller_class.__name__}] PM10 became negative: {state}"
        assert not math.isnan(state), f"[{controller_class.__name__}] PM10 became NaN."


# =====================================================================
# 6. BENCHMARK RUNNER (Executed on `python test_controllers.py`)
# =====================================================================

def run_benchmark():
    print("\n" + "="*75)
    print("🚀 AQI CONTROLLER BENCHMARK INITIALIZATION".center(75))
    print("="*75)
    
    config = EnvironmentConfig()
    generator = DataGenerator(seed=42)
    
    # 1. FAIRNESS FIX: Pre-generate the exact same stochastic noise scenario for all
    scenario_inflows = [generator.traffic_spike(t) for t in range(60)]
    
    # 2. PATH FIX: Safely check both root and models/ folder for the Random Forest
    model_path = "models/rf_aqi_model.pkl" if os.path.exists("models/rf_aqi_model.pkl") else "rf_aqi_model.pkl"
    try:
        from forecaster import AQIForecaster
        forecaster = AQIForecaster(model_path)
        print("[System] Real Random Forest forecaster loaded.")
    except Exception:
        forecaster = DummyForecaster()
        print("[System] Missing RF model. Using DummyForecaster for MPC.")

    controllers = {
        "Threshold (Rule) ": ThresholdController(),
        "PID (Classical)  ": PIDController(kp=1.5, ki=0.08, kd=3.0, integral_limit=40.0, alpha=0.2), # NEW TUNED PID
        "MPC (Predictive) ": MPCController(forecaster=forecaster, horizon=12), # Make sure MPC horizon is 12
        "PPO (Fallback)   ": PPOController(model_path="fake.zip")
    }

    results = {}

    for name, ctrl in controllers.items():
        env = SurrogateEnv(config)
        evaluator = Evaluator()
        
        state = env.reset()
        ctrl.reset()
        
        for t in range(60): 
            inflow = scenario_inflows[t] # Use the shared deterministic scenario
            action = ctrl.choose_action(current_pm10=state, ambient_inflow=inflow)
            
            result = env.step(scrubber_action=action, ambient_pm10_inflow=inflow)
            state = result["next_state"]
            
            evaluator.record(pm10=state, action=action, penalties=result["penalties"])
            
        results[name] = evaluator.summary()

    # Print Summary Table
    print("\n" + "-"*85)
    print(f"{'Controller Pipeline':<20} | {'Avg PM10 (µg/m³)':<15} | {'RMSE (vs 50)':<12} | "
          f"{'Water (kUnits)':<14} | {'Violations'}")
    print("-" * 85)
    
    for name, metrics in results.items():
        avg_pm10 = metrics["Average PM10"]
        rmse = metrics["RMSE"]
        
        # 3. FORMATTING FIX: Scale water usage down so it fits nicely in the table
        water_k = metrics["Water Usage"] / 1000.0 
        vios = metrics["Total Violations"]
        
        print(f"{name:<20} | {avg_pm10:<15.2f} | {rmse:<12.2f} | "
              f"{water_k:<14.1f} | {vios:.2f}")
    
    print("-" * 85)
    print("Benchmark complete. Observe tradeoffs between tracking efficiency and resource usage.\n")
    
if __name__ == "__main__":
    # If file is run directly, execute the benchmark comparison tool
    run_benchmark()