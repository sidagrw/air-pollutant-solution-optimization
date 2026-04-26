import pandas as pd
from pathlib import Path

input_dir = Path("data/raw/cpcb")

train_output = Path("data/processed/train/rvce_train.csv")
test_output = Path("data/processed/test/rvce_test.csv")
combined_output = Path("data/processed/rvce_combined_2023_2025.csv")

train_output.parent.mkdir(parents=True, exist_ok=True)
test_output.parent.mkdir(parents=True, exist_ok=True)
combined_output.parent.mkdir(parents=True, exist_ok=True)

csv_files = sorted(input_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in data/raw/cpcb/")

frames = []

for file in csv_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file)
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

print("Columns found:")
print(df.columns)

# Change "date" below if your actual date column has a different name.
df["date"] = pd.to_datetime(df["Timestamp"], errors="coerce")

df = df.dropna(subset=["date"])
df = df.sort_values("date")

df.to_csv(combined_output, index=False)

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

print("Combined and split complete.")
print(f"Files combined: {len(csv_files)}")
print(f"Total rows: {len(df)}")
print(f"Training rows: {len(train_df)}")
print(f"Testing rows: {len(test_df)}")
print(f"Combined saved to: {combined_output}")
print(f"Train saved to: {train_output}")
print(f"Test saved to: {test_output}")
print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
print("Test date range:", test_df["date"].min(), "to", test_df["date"].max())
