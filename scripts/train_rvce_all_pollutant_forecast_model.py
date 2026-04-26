import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_path = Path("data/processed/train/rvce_train.csv")
test_path = Path("data/processed/test/rvce_test.csv")

model_output = Path("models/trained/rvce_all_pollutant_forecast_random_forest.pkl")
metrics_output = Path("outputs/metrics/rvce_all_pollutant_forecast_metrics.csv")
predictions_output = Path("outputs/predictions/rvce_all_pollutant_forecast_predictions.csv")

model_output.parent.mkdir(parents=True, exist_ok=True)
metrics_output.parent.mkdir(parents=True, exist_ok=True)
predictions_output.parent.mkdir(parents=True, exist_ok=True)

FEATURES = ["AT (°C)", "RH (%)", "WS (m/s)", "RF (mm)", "hour", "month", "day_of_week"]

CANDIDATE_TARGETS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO (µg/m³)",
    "NO2 (µg/m³)",
    "NOx (ppb)",
    "NH3 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "Benzene (µg/m³)",
    "Toluene (µg/m³)",
    "Xylene (µg/m³)",
]


def choose_targets(train_df):
    selected = []
    for col in CANDIDATE_TARGETS:
        if col in train_df.columns:
            valid_count = pd.to_numeric(train_df[col], errors="coerce").notna().sum()
            print(f"{col}: {valid_count} valid rows")
            if valid_count >= 100:
                selected.append(col)

    if not selected:
        raise ValueError("No pollutant columns have enough usable data.")

    print("\nSelected target pollutants:")
    print(selected)
    return selected


def prepare(df, targets, medians=None, label="data"):
    df = df.copy()
    print(f"\nPreparing {label}")
    print("Original rows:", len(df))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    numeric_cols = ["AT (°C)", "RH (%)", "WS (m/s)", "RF (mm)"] + targets

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if medians is None:
        medians = df[numeric_cols].median(numeric_only=True)

    df[numeric_cols] = df[numeric_cols].fillna(medians)

    df["hour"] = df["Timestamp"].dt.hour
    df["month"] = df["Timestamp"].dt.month
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    for target in targets:
        df[f"{target}_next"] = df[target].shift(-1)

    next_targets = [f"{target}_next" for target in targets]

    df = df.dropna(subset=FEATURES + next_targets)

    print("Final usable rows:", len(df))

    X = df[FEATURES]
    y = df[next_targets]

    return df, X, y, medians


def main():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    targets = choose_targets(train_df)

    train_prepared, X_train, y_train, medians = prepare(train_df, targets, label="training data")
    test_prepared, X_test, y_test, _ = prepare(test_df, targets, medians=medians, label="testing data")

    if len(X_train) == 0:
        raise ValueError("Training has 0 rows after cleaning. Check missing values in weather columns.")

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    output_df = test_prepared.copy()

    for i, target in enumerate(targets):
        output_df[f"predicted_{target}"] = predictions[:, i]

    output_df.to_csv(predictions_output, index=False)

    metric_rows = []
    for i, target in enumerate(targets):
        y_true = y_test.iloc[:, i]
        y_pred = predictions[:, i]

        metric_rows.append({
            "target": target,
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "r2": r2_score(y_true, y_pred),
        })

    pd.DataFrame(metric_rows).to_csv(metrics_output, index=False)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "targets": targets,
        },
        model_output,
    )

    print("\nTraining complete.")
    print(pd.DataFrame(metric_rows))
    print("Saved model to:", model_output)
    print("Saved metrics to:", metrics_output)
    print("Saved predictions to:", predictions_output)


if __name__ == "__main__":
    main()
