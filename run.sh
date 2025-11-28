#!/bin/bash
# End-to-end runner for the entire ML pipeline, deployment, and inference

set -e

PROJECT_DIR="/home/sanjaylinux/hpp"
VENV="$PROJECT_DIR/.venv"

echo "=========================================="
echo "ML Pipeline End-to-End Runner"
echo "=========================================="
echo ""

# Activate virtualenv
echo "[1/5] Activating virtualenv..."
source "$VENV/bin/activate"
echo "✓ Virtualenv activated"
echo ""

# Run the main training pipeline
echo "[2/5] Running training pipeline..."
cd "$PROJECT_DIR"
python run_pipeline.py 2>&1 | tail -5
echo "✓ Training pipeline completed"
echo ""

# Run the deployment pipeline
echo "[3/5] Running deployment pipeline..."
python run_deployment.py 2>&1 | tail -5
echo "✓ Deployment pipeline completed"
echo ""

# Run sample prediction
echo "[4/5] Running sample prediction..."
echo "─────────────────────────────────────────"
python sample_predict.py
echo "─────────────────────────────────────────"
echo "✓ Sample prediction completed"
echo ""

# View model artifacts
echo "[5/5] Displaying model artifacts location..."
echo "Model artifacts stored in:"
echo "  /home/sanjaylinux/.config/zenml/local_stores/7bb690eb-01f9-498f-9546-2aec4ac85231/"
echo ""
