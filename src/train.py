import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# Configuration
# ==========================================================
BASE_DIR = "/home/rahul-changezi/mypython/tx_monitoring"
DATASET_PATH = f"{BASE_DIR}/data/raw/synthetic_transactions.csv"

mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
mlflow.set_experiment("Financial_Fraud_Detection")

# ==========================================================
# Load Data
# ==========================================================
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# ==========================================================
# FinBERT
# ==========================================================
print("Loading FinBERT...")
model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_base = AutoModel.from_pretrained(model_name)

print("Generating embeddings...")

tokens = tokenizer(
    list(df["text"]),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = finbert_base(**tokens)
    text_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

# ==========================================================
# Tabular Features
# ==========================================================
raw_tabular = df[
    [
        "tx_amount",
        "account_velocity_1h",
        "ip_class",
        "location_risk"
    ]
].values

scaler = MinMaxScaler()

tabular_features = scaler.fit_transform(raw_tabular)

# ==========================================================
# Feature Fusion
# ==========================================================
X_combined = np.hstack((text_embeddings, tabular_features))
y = df["label"].values

# ==========================================================
# Model
# ==========================================================
class FraudFusionClassifier(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)


model = FraudFusionClassifier(
    input_dim=X_combined.shape[1]
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005
)

inputs = torch.tensor(
    X_combined,
    dtype=torch.float32
)

targets = torch.tensor(
    y,
    dtype=torch.long
)

# ==========================================================
# MLflow Run
# ==========================================================
with mlflow.start_run():

    mlflow.log_param(
        "dataset",
        "synthetic_transactions.csv"
    )

    mlflow.log_param(
        "rows",
        len(df)
    )

    mlflow.log_param(
        "epochs",
        50
    )

    mlflow.log_param(
        "learning_rate",
        0.005
    )

    for epoch in range(50):

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch+1}/50 Loss={loss.item():.4f}"
            )

    mlflow.log_metric(
        "final_loss",
        float(loss.item())
    )

    mlflow.pytorch.log_model(
        model,
        artifact_path="hybrid_model"
    )

print("Training finished.")