import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# Configuration
# ==========================================================
BASE_DIR = "/home/rahul-changezi/mypython/tx_monitoring"

RESULTS_DIR = "results"

PREDICTIONS = f"{RESULTS_DIR}/predictions.csv"

EXPERIMENT_NAME = "Financial_Fraud_Detection"

# ==========================================================
# MLflow
# ==========================================================
mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR}/mlflow.db"
)

mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================================================
# Load predictions
# ==========================================================
print("Loading predictions...")

df = pd.read_csv(PREDICTIONS)

y_true = df["label"]

y_pred = df["prediction"]

# ==========================================================
# Metrics
# ==========================================================
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)

# ==========================================================
# Print
# ==========================================================
print("\nEvaluation Results")
print("---------------------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================================
# Save confusion matrix
# ==========================================================
os.makedirs(RESULTS_DIR, exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.savefig(
    f"{RESULTS_DIR}/confusion_matrix.png"
)

plt.close()

# ==========================================================
# Save report
# ==========================================================
with open(
    f"{RESULTS_DIR}/evaluation_report.txt",
    "w"
) as f:

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")

# ==========================================================
# Log to MLflow
# ==========================================================
with mlflow.start_run(run_name="evaluation"):

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall", recall)

    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact(
        f"{RESULTS_DIR}/confusion_matrix.png"
    )

    mlflow.log_artifact(
        f"{RESULTS_DIR}/evaluation_report.txt"
    )

print("\nEvaluation complete.")