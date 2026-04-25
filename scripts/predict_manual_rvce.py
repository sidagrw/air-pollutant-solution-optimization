import pandas as pd
import joblib
from pathlib import Path

model_path = Path("models/trained/rvce_random_forest.pkl")
train_path = Path("data/processed/train/rvce_train.csv")

model = joblib.load(model_path)
train_df = pd.read_csv(train_path)

numeric_columns = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
    "RF (mm)",
]

for col in numeric_columns:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

defaults = train_df[numeric_columns].median(numeric_only=True)

pm25 = float(input("Enter PM2.5 value: "))
no2 = float(input("Enter NO2 value: "))
temperature = float(input("Enter temperature, AT °C: "))
humidity = float(input("Enter relative humidity, RH %: "))
wind_speed = float(input("Enter wind speed, WS m/s: "))
hour = int(input("Enter hour, 0-23: "))
month = int(input("Enter month, 1-12: "))
day_of_week = int(input("Enter day of week, Monday=0, Sunday=6: "))

sample = {
    "PM2.5 (µg/m³)": pm25,
    "PM10 (µg/m³)": defaults["PM10 (µg/m³)"],
    "NO2 (µg/m³)": no2,
    "SO2 (µg/m³)": defaults["SO2 (µg/m³)"],
    "CO (mg/m³)": defaults["CO (mg/m³)"],
    "Ozone (µg/m³)": defaults["Ozone (µg/m³)"],
    "AT (°C)": temperature,
    "RH (%)": humidity,
    "WS (m/s)": wind_speed,
    "RF (mm)": defaults["RF (mm)"],
    "hour": hour,
    "month": month,
    "day_of_week": day_of_week,
}

X = pd.DataFrame([sample])
prediction = model.predict(X)[0]

print()
print(f"Predicted next AQI proxy: {prediction:.2f}")

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

print(f"Approximate AQI category: {category}")
