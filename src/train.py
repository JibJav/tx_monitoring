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

DATASET_PATH = "data/processed/transactions_processed.csv"

EXPERIMENT_NAME = "Financial_Fraud_Detection"

EPOCHS = 50
LEARNING_RATE = 0.005
HIDDEN_DIM = 64

MODEL_NAME = "ProsusAI/finbert"

# ==========================================================
# MLflow
# ==========================================================
mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR}/mlflow.db"
)

mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================================================
# Load Dataset
# ==========================================================
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Loaded {len(df)} transactions.")

# Better run names in MLflow
run_name = (
    f"rows_{len(df)}"
    f"_fraud_{df['label'].mean():.2f}"
    f"_lr_{LEARNING_RATE}"
)

# ==========================================================
# FinBERT
# ==========================================================
print("Loading FinBERT...")

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

    text_embeddings = (
        outputs.last_hidden_state[:, 0, :]
        .numpy()
    )

# ==========================================================
# Tabular Features
# ==========================================================
FEATURE_COLUMNS = [
    "tx_amount",
    "account_velocity_1h",
    "ip_class",
    "location_risk",
    "high_amount_flag",
    "velocity_score"
]

raw_features = df[
    FEATURE_COLUMNS
].values

scaler = MinMaxScaler()

tabular_features = scaler.fit_transform(
    raw_features
)

# ==========================================================
# Feature Fusion
# ==========================================================
X = np.hstack(
    (
        text_embeddings,
        tabular_features
    )
)

y = df["label"].values

# ==========================================================
# Model
# ==========================================================
class FraudFusionClassifier(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                2
            )
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
# Training + MLflow
# ==========================================================
with mlflow.start_run(run_name=run_name):

    # Parameters
    mlflow.log_param("dataset", DATASET_PATH)
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("hidden_dim", HIDDEN_DIM)
    mlflow.log_param("base_model", MODEL_NAME)
    mlflow.log_param("feature_columns", FEATURE_COLUMNS)

    # Dataset metrics
    mlflow.log_metric("fraud_ratio", float(df["label"].mean()))
    mlflow.log_metric("avg_tx_amount", float(df["tx_amount"].mean()))
    mlflow.log_metric("max_tx_amount", float(df["tx_amount"].max()))

    # ==========================
    # Training loop
    # ==========================
    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = criterion(predictions, targets)

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} Loss={loss.item():.4f}")

    # Final metric
    mlflow.log_metric(
        "final_loss",
        float(loss.item())
    )

    # ==========================================================
    # Save predictions
    # ==========================================================
    with torch.no_grad():

        logits = model(inputs)

        predicted_classes = torch.argmax(
            logits,
            dim=1
        ).numpy()

    df["prediction"] = predicted_classes

    df.to_csv(
        "results/predictions.csv",
        index=False
    )

    # ==========================================================
    # Save model to MLflow
    # ==========================================================
    mlflow.pytorch.log_model(
        model,
        name="hybrid_model"
    )

print("\nTraining completed successfully.")