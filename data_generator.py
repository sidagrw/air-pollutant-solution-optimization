# data_generator.py

import numpy as np

class DataGenerator:
    """
    Generates synthetic environmental disturbances
    for AQI simulation experiments.
    """

    def __init__(self, seed=42):

        np.random.seed(seed)

    def normal_traffic(self, t):

        base = 10
        noise = np.random.normal(0, 2)

        return max(0, base + noise)

    def traffic_spike(self, t):
        base = 12.0 # Slightly higher baseline 
        
        # Rush hour spike (make it slightly shorter, but sharp)
        if 20 <= t <= 28:
            base += 32.0 # Total inflow = 44.0 (Pushes system into actuator saturation)
            
        # Add stochastic environmental noise
        noise = np.random.normal(0, 2.5)
        
        return max(0.0, base + noise)

    def pollution_burst(self, t):

        base = 8

        # Sudden industrial event
        if t == 50:
            base += 80

        return base

    def windy_conditions(self, t):

        # Wind reduces effective inflow
        wind_strength = np.random.uniform(0, 10)

        pollution = 15 - wind_strength

        return max(0, pollution)

    def low_water_scenario(self, t):

        base = 12

        # Used later to constrain MPC actions
        return base