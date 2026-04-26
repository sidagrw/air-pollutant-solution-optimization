import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


train_path = Path("data/processed/train/rvce_train.csv")
test_path = Path("data/processed/test/rvce_test.csv")

model_output = Path("models/trained/rvce_random_forest.pkl")
metrics_output = Path("outputs/metrics/rvce_random_forest_metrics.csv")
predictions_output = Path("outputs/predictions/rvce_predictions.csv")

model_output.parent.mkdir(parents=True, exist_ok=True)
metrics_output.parent.mkdir(parents=True, exist_ok=True)
predictions_output.parent.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
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


def compute_aqi(row):
    return max(
        pm25_aqi(row["PM2.5 (µg/m³)"]),
        pm10_aqi(row["PM10 (µg/m³)"]),
        no2_aqi(row["NO2 (µg/m³)"]),
    )


def prepare(df, medians=None, label="data"):
    df = df.copy()

    print(f"\nPreparing {label}")
    print("Original rows:", len(df))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    print("Rows after valid Timestamp:", len(df))

    numeric_cols = [
        "PM2.5 (µg/m³)",
        "PM10 (µg/m³)",
        "NO2 (µg/m³)",
        "AT (°C)",
        "RH (%)",
        "WS (m/s)",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nMissing values before filling:")
    print(df[numeric_cols].isna().sum())

    if medians is None:
        medians = df[numeric_cols].median(numeric_only=True)

    df[numeric_cols] = df[numeric_cols].fillna(medians)

    df = df.dropna(subset=numeric_cols)

    print("Rows after filling missing numeric values:", len(df))

    df["hour"] = df["Timestamp"].dt.hour
    df["month"] = df["Timestamp"].dt.month
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    df["aqi"] = df.apply(compute_aqi, axis=1)
    df["aqi_next"] = df["aqi"].shift(-1)
    df = df.dropna(subset=["aqi_next"])

    print("Final usable rows:", len(df))

    X = df[FEATURES]
    y = df["aqi_next"]

    return df, X, y, medians


def main():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_prepared, X_train, y_train, medians = prepare(train_df, label="training data")
    test_prepared, X_test, y_test, _ = prepare(test_df, medians=medians, label="testing data")

    if len(X_train) == 0:
        raise ValueError("Training data has 0 usable rows. Check pollutant columns and timestamp parsing.")

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame([{
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features_used": ", ".join(FEATURES),
    }])

    metrics.to_csv(metrics_output, index=False)

    output_df = test_prepared.copy()
    output_df["actual_aqi_next"] = y_test.values
    output_df["predicted_aqi_next"] = predictions
    output_df.to_csv(predictions_output, index=False)

    joblib.dump(model, model_output)

    print("\nRandom Forest training complete.")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("Model saved to:", model_output)
    print("Metrics saved to:", metrics_output)
    print("Predictions saved to:", predictions_output)


if __name__ == "__main__":
    main()
