import mlflow
import mlflow.pytorch
import torch

import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
# ==========================================================
# Configuration
# ==========================================================
BASE_DIR = "/home/rahul-changezi/mypython/tx_monitoring"
DATASET_PATH = f"{BASE_DIR}/data/raw/synthetic_transactions.csv"

EXPERIMENT_NAME = "Financial_Fraud_Detection"

# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.005
HIDDEN_DIM = 64

# ==========================================================
# MLflow
# ==========================================================
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================================================
# Load Dataset
# ==========================================================
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print(f"Loaded {len(df)} transactions.")

# ==========================================================
# FinBERT
# ==========================================================
print("Loading FinBERT...")

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
finbert = AutoModel.from_pretrained(MODEL_NAME)

print("Generating embeddings...")

tokens = tokenizer(
    list(df["text"]),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = finbert(**tokens)
    text_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

# ==========================================================
# Tabular Features
# ==========================================================
FEATURE_COLUMNS = [
    "tx_amount",
    "account_velocity_1h",
    "ip_class",
    "location_risk"
]

raw_features = df[FEATURE_COLUMNS].values

scaler = MinMaxScaler()

tabular_features = scaler.fit_transform(raw_features)

# ==========================================================
# Feature Fusion
# ==========================================================
X = np.hstack((text_embeddings, tabular_features))

y = df["label"].values

# ==========================================================
# Model
# ==========================================================
class FraudFusionClassifier(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2)
        )

    def forward(self, x):
        return self.network(x)


model = FraudFusionClassifier(
    input_dim=X.shape[1]
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

inputs = torch.tensor(
    X,
    dtype=torch.float32
)

targets = torch.tensor(
    y,
    dtype=torch.long
)

# ==========================================================
# Training
# ==========================================================
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

with mlflow.start_run(run_name=run_name):

    mlflow.log_param("dataset", DATASET_PATH)
    mlflow.log_param("rows", len(df))
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("hidden_dim", HIDDEN_DIM)
    mlflow.log_param("base_model", MODEL_NAME)

    mlflow.log_metric(
        "fraud_ratio",
        float(df["label"].mean())
    )
    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} Loss={loss.item():.4f}"
            )

    mlflow.log_metric(
        "final_loss",
        float(loss.item())
    )

    mlflow.pytorch.log_model(
        model,
        artifact_path="hybrid_model"
    )

print("\nTraining complete.")