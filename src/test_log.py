import mlflow

# 1. Verbindung zur SQLite-Datenbank herstellen (WICHTIG!)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Ein Experiment auswählen oder erstellen
mlflow.set_experiment("Mein_Erstes_Experiment")

# 3. Den Run starten und Daten loggen
with mlflow.start_run():
    # Parameter loggen (z. B. Hyperparameter)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)
    
    # Metriken loggen (z. B. Trainingsergebnisse)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    
    print("Run erfolgreich in sqlite:///mlflow.db geloggt!")
