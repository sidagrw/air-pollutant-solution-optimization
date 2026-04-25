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


def clean_number_column(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calculate_simple_aqi(row):
    """
    Simple AQI proxy using major pollutants.
    This is not the official CPCB breakpoint formula.
    It creates a usable AQI-like target for model training.
    """
    pm25 = row["PM2.5 (µg/m³)"]
    pm10 = row["PM10 (µg/m³)"]
    no2 = row["NO2 (µg/m³)"]
    co = row["CO (mg/m³)"]
    ozone = row["Ozone (µg/m³)"]

    pm25_index = pm25 * 2
    pm10_index = pm10
    no2_index = no2 * 1.25
    co_index = co * 50
    ozone_index = ozone

    return max(pm25_index, pm10_index, no2_index, co_index, ozone_index)


def prepare_data(df):
    df = df.copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

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
        df = clean_number_column(df, col)

    df = df.dropna(subset=numeric_columns)

    df["hour"] = df["Timestamp"].dt.hour
    df["month"] = df["Timestamp"].dt.month
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    df["aqi_proxy"] = df.apply(calculate_simple_aqi, axis=1)

    df["aqi_next"] = df["aqi_proxy"].shift(-1)

    df = df.dropna(subset=["aqi_next"])

    features = numeric_columns + ["hour", "month", "day_of_week"]

    X = df[features]
    y = df["aqi_next"]

    return df, X, y, features


def main():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df, X_train, y_train, features = prepare_data(train_df)
    test_df, X_test, y_test, _ = prepare_data(test_df)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
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
        "features_used": ", ".join(features),
    }])

    metrics.to_csv(metrics_output, index=False)

    output_df = test_df.copy()
    output_df["actual_aqi_next"] = y_test.values
    output_df["predicted_aqi_next"] = predictions
    output_df.to_csv(predictions_output, index=False)

    joblib.dump(model, model_output)

    print("Random Forest training complete.")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("Model saved to:", model_output)
    print("Metrics saved to:", metrics_output)
    print("Predictions saved to:", predictions_output)


if __name__ == "__main__":
    main()
