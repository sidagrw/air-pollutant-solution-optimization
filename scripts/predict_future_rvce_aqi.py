import pandas as pd
import joblib
from pathlib import Path

model_path = Path("models/trained/rvce_forecast_random_forest.pkl")

model = joblib.load(model_path)

date_input = input("Enter future date, YYYY-MM-DD: ")
hour = int(input("Enter hour, 0-23: "))
temperature = float(input("Enter forecast temperature, AT °C: "))
humidity = float(input("Enter forecast relative humidity, RH %: "))
wind_speed = float(input("Enter forecast wind speed, WS m/s: "))
rainfall = float(input("Enter forecast rainfall, RF mm: "))

date = pd.to_datetime(date_input)

sample = {
    "AT (°C)": temperature,
    "RH (%)": humidity,
    "WS (m/s)": wind_speed,
    "RF (mm)": rainfall,
    "hour": hour,
    "month": date.month,
    "day_of_week": date.dayofweek,
}

X = pd.DataFrame([sample])
prediction = model.predict(X)[0]

print()
print(f"Predicted AQI near RVCE: {prediction:.2f}")

if prediction <= 50:
    category = "Good"
elif prediction <= 100:
    category = "Satisfactory"
elif prediction <= 200:
    category = "Moderate"
elif prediction <= 300:
    category = "Poor"
elif prediction <= 400:
    category = "Very Poor"
else:
    category = "Severe"

print(f"Predicted AQI category: {category}")
