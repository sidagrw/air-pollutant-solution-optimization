import pandas as pd
import joblib
from pathlib import Path

model_path = Path("models/trained/rvce_pollutant_forecast_random_forest.pkl")
model = joblib.load(model_path)

FEATURES = [
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
    "RF (mm)",
    "hour",
    "month",
    "day_of_week",
]


def sub_index(c, bp_lo, bp_hi, i_lo, i_hi):
    return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c - bp_lo) + i_lo


def pm25_aqi(x):
    if x <= 30: return sub_index(x, 0, 30, 0, 50)
    elif x <= 60: return sub_index(x, 31, 60, 51, 100)
    elif x <= 90: return sub_index(x, 61, 90, 101, 200)
    elif x <= 120: return sub_index(x, 91, 120, 201, 300)
    elif x <= 250: return sub_index(x, 121, 250, 301, 400)
    else: return sub_index(x, 251, 500, 401, 500)


def pm10_aqi(x):
    if x <= 50: return sub_index(x, 0, 50, 0, 50)
    elif x <= 100: return sub_index(x, 51, 100, 51, 100)
    elif x <= 250: return sub_index(x, 101, 250, 101, 200)
    elif x <= 350: return sub_index(x, 251, 350, 201, 300)
    elif x <= 430: return sub_index(x, 351, 430, 301, 400)
    else: return sub_index(x, 431, 600, 401, 500)


def no2_aqi(x):
    if x <= 40: return sub_index(x, 0, 40, 0, 50)
    elif x <= 80: return sub_index(x, 41, 80, 51, 100)
    elif x <= 180: return sub_index(x, 81, 180, 101, 200)
    elif x <= 280: return sub_index(x, 181, 280, 201, 300)
    elif x <= 400: return sub_index(x, 281, 400, 301, 400)
    else: return sub_index(x, 401, 1000, 401, 500)


def category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


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

X = pd.DataFrame([sample], columns=FEATURES)
prediction = model.predict(X)[0]

pm25 = max(prediction[0], 0)
pm10 = max(prediction[1], 0)
no2 = max(prediction[2], 0)

pm25_index = pm25_aqi(pm25)
pm10_index = pm10_aqi(pm10)
no2_index = no2_aqi(no2)

subindices = {
    "PM2.5": pm25_index,
    "PM10": pm10_index,
    "NO2": no2_index,
}

dominant_pollutant = max(subindices, key=subindices.get)
final_aqi = subindices[dominant_pollutant]

print()
print("Predicted pollutant concentrations near RVCE:")
print(f"PM2.5: {pm25:.2f} µg/m³")
print(f"PM10: {pm10:.2f} µg/m³")
print(f"NO2: {no2:.2f} µg/m³")

print()
print("Predicted AQI sub-indices:")
print(f"PM2.5 sub-index: {pm25_index:.2f}")
print(f"PM10 sub-index: {pm10_index:.2f}")
print(f"NO2 sub-index: {no2_index:.2f}")

print()
print(f"Final predicted AQI: {final_aqi:.2f}")
print(f"Dominant pollutant: {dominant_pollutant}")
print(f"AQI category: {category(final_aqi)}")
