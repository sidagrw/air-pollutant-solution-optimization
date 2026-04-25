import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

train_path = Path("data/processed/train/rvce_train.csv")
test_path = Path("data/processed/test/rvce_test.csv")

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["Timestamp"] = pd.to_datetime(train["Timestamp"], errors="coerce")
test["Timestamp"] = pd.to_datetime(test["Timestamp"], errors="coerce")

pollutants = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
]

for col in pollutants:
    train[col] = pd.to_numeric(train[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")

    plt.figure(figsize=(12, 6))
    plt.plot(train["Timestamp"], train[col], label="Training Data")
    plt.plot(test["Timestamp"], test[col], label="Test Data")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.title(f"Train vs Test Comparison: {col}")
    plt.legend()
    plt.tight_layout()

    safe_name = (
        col.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("µ", "u")
        .replace("³", "3")
        .replace("%", "percent")
        .replace("°", "deg")
    )

    output_path = output_dir / f"train_test_{safe_name}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")

print("All graphs generated.")
