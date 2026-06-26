import pandas as pd
import os

RAW_PATH = "data/raw/synthetic_transactions.csv"
OUTPUT_PATH = "data/processed/transactions_processed.csv"

print("Loading raw dataset...")

df = pd.read_csv(RAW_PATH)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values
df.fillna(0, inplace=True)

# Feature engineering
df["high_amount_flag"] = (
    df["tx_amount"] > 10000
).astype(int)

df["velocity_score"] = (
    df["account_velocity_1h"] * df["location_risk"]
)

# Create output folder
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Processed dataset saved.")