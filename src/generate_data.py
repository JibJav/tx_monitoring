import pandas as pd
import numpy as np
import random
import os

# ======================================
# Experiment Parameters
# ======================================
SEED = 42

NUM_RECORDS = 10000
FRAUD_RATIO = 0.35
STEALTH_FRAUD_RATIO = 0.35

CLEAN_AMOUNT_MIN = 5
CLEAN_AMOUNT_MAX = 3000

FRAUD_AMOUNT_MIN = 8000
FRAUD_AMOUNT_MAX = 95000

OUTPUT_FILE = "data/raw/synthetic_transactions.csv"

# ======================================
# Seed
# ======================================
np.random.seed(SEED)
random.seed(SEED)

print(f"Generating {NUM_RECORDS} transactions...")

# ======================================
# Transaction Texts
# ======================================
clean_texts = [
    "Salary payment received",
    "Monthly rent transfer",
    "Electricity bill payment",
    "Amazon online purchase",
    "Restaurant payment",
    "Subscription renewal",
    "P2P transfer to friend"
]

fraud_texts = [
    "Urgent cash out offshore account",
    "Crypto wallet transfer",
    "International private liquidation",
    "Temporary holding account routing",
    "High risk swift transfer"
]

stealth_fraud_texts = [
    "Gift for mother",
    "Dinner split",
    "Loan repayment",
    "Services rendered payment"
]

# ======================================
# Generate rows
# ======================================
rows = []

for i in range(NUM_RECORDS):

    tx_id = 100000 + i

    is_fraud = 1 if random.random() < FRAUD_RATIO else 0

    if is_fraud == 0:

        text = random.choice(clean_texts)

        tx_amount = round(
            random.uniform(
                CLEAN_AMOUNT_MIN,
                CLEAN_AMOUNT_MAX
            ),
            2
        )

        velocity = random.choice([1.0, 2.0])

        ip_class = 0

        location_risk = (
            1 if random.random() < 0.05 else 0
        )

    else:

        if random.random() < STEALTH_FRAUD_RATIO:

            text = random.choice(
                stealth_fraud_texts
            )

        else:

            text = random.choice(
                fraud_texts
            )

        tx_amount = round(
            random.uniform(
                FRAUD_AMOUNT_MIN,
                FRAUD_AMOUNT_MAX
            ),
            2
        )

        velocity = random.randint(5, 15)

        ip_class = (
            1 if random.random() < 0.80 else 0
        )

        location_risk = (
            1 if random.random() < 0.70 else 0
        )

    rows.append(
        [
            tx_id,
            text,
            tx_amount,
            velocity,
            ip_class,
            location_risk,
            is_fraud
        ]
    )

# ======================================
# DataFrame
# ======================================
df = pd.DataFrame(
    rows,
    columns=[
        "transaction_id",
        "text",
        "tx_amount",
        "account_velocity_1h",
        "ip_class",
        "location_risk",
        "label"
    ]
)

# ======================================
# Save
# ======================================
os.makedirs("data/raw", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(df["label"].value_counts())
print(f"Saved to {OUTPUT_FILE}")