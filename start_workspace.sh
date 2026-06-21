#!/bin/bash

############################################
# TX Monitoring Workspace Startup Script
############################################

echo "==========================================="
echo "Starting TX Monitoring Workspace..."
echo "==========================================="

# Project directory
PROJECT_DIR=~/mypython/tx_monitoring

# Activate virtualenv
cd $PROJECT_DIR || exit
source testenv/bin/activate

echo "Virtual environment activated."

############################################
# MLflow
############################################
echo "Starting MLflow..."

mkdir -p logs

nohup mlflow ui \
    --host 0.0.0.0 \
    --port 5000 \
    > logs/mlflow.log 2>&1 &

############################################
# Jupyter Notebook
############################################
echo "Starting Jupyter Notebook..."

nohup jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8888 \
    > logs/jupyter.log 2>&1 &
# run "jupyter server list"

############################################
# DVC
############################################
echo "Checking DVC status..."

dvc status

############################################
# Feast Feature Store
############################################
echo "Feature repository:"

cd fraud_feature_store/feature_repo || exit

feast registry-dump > /dev/null

cd $PROJECT_DIR

############################################
# Dashboard Links
############################################

echo ""
echo "==========================================="
echo "Workspace Started"
echo "==========================================="
echo ""
echo "# MLflow Dashboard"
echo "http://localhost:5000"
echo ""
echo "# Jupyter Notebook"
echo "http://localhost:8888"
echo ""
echo "# DVC status"
echo "dvc status"
echo ""
echo "# Feast feature repo"
echo "$PROJECT_DIR/fraud_feature_store/feature_repo"
echo ""
echo "# Stop MLflow"
echo "pkill -f mlflow"
echo ""
echo "# Stop Jupyter"
echo "pkill -f jupyter"
echo ""