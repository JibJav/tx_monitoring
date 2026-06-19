import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler

# 1. Point to our absolute database path
BASE_DIR = "/home/rahul-changezi/mypython/tx_monitoring"
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
mlflow.set_experiment("Financial_Fraud_Detection_v2")
mlflow.start_run()

print("Step 1: Loading the 1,500 synthetic transactions from CSV...")
df = pd.read_csv(f"{BASE_DIR}/src/synthetic_transactions.csv")

print("Step 2: Initializing FinBERT Text Extractor...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_base = AutoModel.from_pretrained(model_name)

# Extracting text embeddings batch-by-batch to save computer memory
print("Step 3: Extracting text embeddings (this might take a minute)...")
tokens = tokenizer(list(df["text"]), padding=True, truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    outputs = finbert_base(**tokens)
    text_embeddings = outputs.last_hidden_state[:, 0, :].numpy() # Shape: (1500, 768)

print("Step 4: Applying Feature Scaling to Feature Store metrics...")
raw_tabular = df[["tx_amount", "account_velocity_1h", "ip_class", "location_risk"]].values

# MinMaxScaler shrinks all numbers down to play fair between 0 and 1
scaler = MinMaxScaler()
tabular_features = scaler.fit_transform(raw_tabular) # Shape: (1500, 4)

# FUSE THEM: 768 text columns + 4 math columns = 772 dimensions total
X_combined = np.hstack((text_embeddings, tabular_features))
Y_labels = df["label"].values

print("Step 5: Training the Multimodal Fusion Classifier...")
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

hybrid_model = FraudFusionClassifier(input_dim=X_combined.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.005)

# Convert arrays to PyTorch Tensors
inputs = torch.tensor(X_combined, dtype=torch.float32)
targets = torch.tensor(Y_labels, dtype=torch.long)

# Run 50 epochs over the large dataset
for epoch in range(50):
    optimizer.zero_grad()
    outputs = hybrid_model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50] -> Current Loss: {loss.item():.4f}")

# Log all system details to MLflow Dashboard
mlflow.log_param("base_text_model", "ProsusAI/finbert")
mlflow.log_param("dataset_size", len(df))
mlflow.log_param("feature_store_columns", ["tx_amount", "account_velocity_1h", "ip_class", "location_risk"])
mlflow.log_metric("final_loss", float(loss.item()))

print("Step 6: Saving our smart Multimodal Model to MLflow...")
mlflow.pytorch.log_model(hybrid_model, artifact_path="hybrid_fraud_model")

run_id = mlflow.active_run().info.run_id
print("\n=======================================================")
print(f"Success! Use this New Run ID inside your main.py: {run_id}")
print("=======================================================")
mlflow.end_run()
