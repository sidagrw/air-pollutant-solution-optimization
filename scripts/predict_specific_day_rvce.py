import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models/trained/rvce_all_pollutant_forecast_random_forest.pkl")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
TARGETS = bundle["targets"]

date_input = input("Enter date, YYYY-MM-DD: ")
hour = int(input("Enter hour, 0-23: "))
temperature = float(input("Enter temperature, AT °C: "))
humidity = float(input("Enter relative humidity, RH %: "))
wind_speed = float(input("Enter wind speed, WS m/s: "))
rainfall = float(input("Enter rainfall, RF mm: "))

date = pd.to_datetime(date_input)

X = pd.DataFrame([{
    "AT (°C)": temperature,
    "RH (%)": humidity,
    "WS (m/s)": wind_speed,
    "RF (mm)": rainfall,
    "hour": hour,
    "month": date.month,
    "day_of_week": date.dayofweek,
}], columns=FEATURES)

predictions = model.predict(X)[0]

print()
print("Predicted pollutant concentrations near RVCE:")
for target, value in zip(TARGETS, predictions):
    print(f"{target}: {max(value, 0):.2f}")
