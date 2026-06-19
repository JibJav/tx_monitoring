from fastapi import FastAPI
import mlflow.pytorch
import torch
import numpy as np
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI(title="Production Multimodal Fraud Detection API")

# 1. Load the core model infrastructure
print("Loading FinBERT infrastructure...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_base = AutoModel.from_pretrained(model_name)

# 2. Set absolute database tracking path
BASE_DIR = "/home/rahul-changezi/mypython/tx_monitoring"
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")

# 3. Load your model from MLflow
MODEL_URI = "runs:/d3c7c27de06d4e46a8044bdff702fc00/hybrid_fraud_model"
print(f"Loading hybrid classification weights from: {MODEL_URI}")
hybrid_classifier = mlflow.pytorch.load_model(model_uri=MODEL_URI)
hybrid_classifier.eval()

# ADDED FOR PRODUCTION ROBUSTNESS: 
# Training data minimum and maximum baselines for scaling our 4 metrics
# Order matches: [tx_amount, account_velocity_1h, ip_class, location_risk]
TABULAR_MIN = np.array([5.0, 1.0, 0.0, 0.0])
TABULAR_MAX = np.array([95000.0, 15.0, 1.0, 1.0])

class TransactionRequest(BaseModel):
    user_id: int
    description: str
    current_tx_amount: float
    ip_class: str        
    location: str        

@app.post("/predict")
def predict(tx: TransactionRequest):
    # Simulating a live Feature Store lookup based on incoming profile metrics
    live_velocity = 1.0 if tx.ip_class == "normal" else 12.0 
    
    ip_numeric = 1 if tx.ip_class == "suspicious" else 0
    location_numeric = 1 if tx.location != "US" else 0 
    
    # Extract text features using FinBERT
    tokens = tokenizer(tx.description, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = finbert_base(**tokens)
        text_embedding = outputs.last_hidden_state[:, 0, :].numpy() 

    # Gather unscaled tabular metrics
    raw_tabular = np.array([[tx.current_tx_amount, live_velocity, ip_numeric, location_numeric]])
    
    # CHANGED FOR PRODUCTION READYNESS: Live Feature Scaling Engine
    # This squeezes incoming raw values between 0 and 1 so they match what the model learned!
    tabular_features = (raw_tabular - TABULAR_MIN) / (TABULAR_MAX - TABULAR_MIN + 1e-8)

    # Fuse text embeddings with the scaled tabular feature metrics
    X_combined = np.hstack((text_embedding, tabular_features))
    
    # Run the model
    with torch.no_grad():
        tensor_input = torch.tensor(X_combined, dtype=torch.float32)
        raw_outputs = hybrid_classifier(tensor_input)
        probabilities = torch.softmax(raw_outputs, dim=1).numpy()[0]
        prediction_class = int(np.argmax(probabilities))

    return {
        "user_id": tx.user_id,
        "prediction": "Fraudulent" if prediction_class == 1 else "Clean",
        "fraud_risk_probability": round(float(probabilities[1]), 4),
        "clean_probability": round(float(probabilities[0]), 4)
    }
