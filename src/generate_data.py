import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

num_records = 1500
print(f"Generating {num_records} production-style fraud records...")

# 1. Base Text Templates
clean_texts = [
    "Salary payment received from corporate payroll",
    "Monthly rent transfer for apartment",
    "Reimbursement for office supplies and lunch",
    "P2P transfer to friend for dinner split",
    "Utility bill payment for electricity",
    "Online shopping checkout marketplace",
    "Subscription renewal for streaming service"
]

fraud_texts = [
    "Urgent cash out to offshore account",
    "Divide amounts to avoid tax control structure",
    "Crypto wallet load high risk transfer",
    "Temporary holding account routing swift",
    "Immediate international private liquidation"
]

# Tricky fake descriptions used by fraudsters to try and trick the NLP model
stealth_fraud_texts = [
    "Gift for mom birthday",
    "Dinner split with family",
    "Loan repayment to cousin",
    "Services rendered payment"
]

data = []

for i in range(num_records):
    tx_id = 100000 + i
    
    # Flip a coin: 85% Clean transactions, 15% Fraudulent
    is_fraud = 1 if random.random() < 0.15 else 0
    
    if is_fraud == 0:
        # CLEAN PROFILE
        text = random.choice(clean_texts)
        tx_amount = round(random.uniform(5.0, 3000.0), 2)       # Normal, lower amounts
        velocity = float(random.choice([1.0, 2.0]))             # Low transaction speed
        ip_class = 0                                           # Safe IP address
        location_risk = 1 if random.random() < 0.05 else 0     # Rarely traveling
        
    else:
        # FRAUD PROFILE
        # 30% of fraudsters use a clever, fake description to try and fool FinBERT!
        if random.random() < 0.30:
            text = random.choice(stealth_fraud_texts)
        else:
            text = random.choice(fraud_texts)
            
        tx_amount = round(random.uniform(8000.0, 95000.0), 2)  # High amounts
        velocity = float(random.randint(5, 15))                # Rapid firing transactions
        ip_class = 1 if random.random() < 0.80 else 0          # Mostly suspicious proxies
        location_risk = 1 if random.random() < 0.70 else 0     # Mostly overseas routing

    data.append([tx_id, text, tx_amount, velocity, ip_class, location_risk, is_fraud])

# 2. Build DataFrame
df = pd.DataFrame(data, columns=[
    "transaction_id", "text", "tx_amount", 
    "account_velocity_1h", "ip_class", "location_risk", "label"
])

# Ensure the src directory exists
os.makedirs("src", exist_ok=True)

# Save it to your folder
df.to_csv("src/synthetic_transactions.csv", index=False)
print("Success! Dataset saved to 'src/synthetic_transactions.csv'.")
print(df["label"].value_counts())
