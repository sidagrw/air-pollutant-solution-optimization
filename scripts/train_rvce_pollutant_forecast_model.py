import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_path = Path("data/processed/train/rvce_train.csv")
test_path = Path("data/processed/test/rvce_test.csv")

model_output = Path("models/trained/rvce_pollutant_forecast_random_forest.pkl")
metrics_output = Path("outputs/metrics/rvce_pollutant_forecast_metrics.csv")
predictions_output = Path("outputs/predictions/rvce_pollutant_forecast_predictions.csv")

model_output.parent.mkdir(parents=True, exist_ok=True)
metrics_output.parent.mkdir(parents=True, exist_ok=True)
predictions_output.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
    "RF (mm)",
    "hour",
    "month",
    "day_of_week",
]

TARGETS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
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


def compute_aqi_from_pollutants(pm25, pm10, no2):
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

    return final_aqi, dominant_pollutant, subindices


def prepare(df, medians=None, label="data"):
    df = df.copy()

    print(f"\nPreparing {label}")
    print("Original rows:", len(df))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    numeric_cols = TARGETS + [
        "AT (°C)",
        "RH (%)",
        "WS (m/s)",
        "RF (mm)",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if medians is None:
        medians = df[numeric_cols].median(numeric_only=True)

    df[numeric_cols] = df[numeric_cols].fillna(medians)
    df = df.dropna(subset=numeric_cols)

    df["hour"] = df["Timestamp"].dt.hour
    df["month"] = df["Timestamp"].dt.month
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    # Predict next time-step pollutant concentrations.
    for target in TARGETS:
        df[f"{target}_next"] = df[target].shift(-1)

    next_targets = [f"{target}_next" for target in TARGETS]
    df = df.dropna(subset=next_targets)

    X = df[FEATURES]
    y = df[next_targets]

    print("Final usable rows:", len(df))

    return df, X, y, medians


def main():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_prepared, X_train, y_train, medians = prepare(train_df, label="training data")
    test_prepared, X_test, y_test, _ = prepare(test_df, medians=medians, label="testing data")

    base_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    prediction_df = test_prepared.copy()

    for i, target in enumerate(TARGETS):
        prediction_df[f"predicted_{target}"] = predictions[:, i]

    aqi_values = []
    dominant_pollutants = []
    pm25_subindices = []
    pm10_subindices = []
    no2_subindices = []

    for row in predictions:
        pm25, pm10, no2 = row
        aqi, dominant, subindices = compute_aqi_from_pollutants(pm25, pm10, no2)

        aqi_values.append(aqi)
        dominant_pollutants.append(dominant)
        pm25_subindices.append(subindices["PM2.5"])
        pm10_subindices.append(subindices["PM10"])
        no2_subindices.append(subindices["NO2"])

    prediction_df["predicted_aqi"] = aqi_values
    prediction_df["dominant_pollutant"] = dominant_pollutants
    prediction_df["predicted_pm25_subindex"] = pm25_subindices
    prediction_df["predicted_pm10_subindex"] = pm10_subindices
    prediction_df["predicted_no2_subindex"] = no2_subindices

    prediction_df.to_csv(predictions_output, index=False)

    metric_rows = []

    for i, target in enumerate(TARGETS):
        y_true = y_test.iloc[:, i]
        y_pred = predictions[:, i]

        metric_rows.append({
            "target": target,
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "r2": r2_score(y_true, y_pred),
        })

    pd.DataFrame(metric_rows).to_csv(metrics_output, index=False)

    joblib.dump(model, model_output)

    print("\nPollutant forecast model training complete.")
    print(pd.DataFrame(metric_rows))
    print("Model saved to:", model_output)
    print("Metrics saved to:", metrics_output)
    print("Predictions saved to:", predictions_output)


if __name__ == "__main__":
    main()
