# forecaster.py

import joblib
import numpy as np

class AQIForecaster:
    """
    Random Forest PM10 forecaster.
    """

    def __init__(self, model_path):

        self.model = joblib.load(model_path)

    def predict(self, current_pm10,
                      ambient_inflow,
                      scrubber_action):

        features = np.array([[
            current_pm10,
            ambient_inflow,
            scrubber_action
        ]])

        prediction = self.model.predict(features)

        return prediction[0]