#!/bin/bash

# Complete HPP Project Runner
# This script runs the entire ML pipeline end-to-end:
# 1. Activates the virtual environment
# 2. Runs the training pipeline
# 3. Runs the deployment service
# 4. Runs sample predictions
# 5. Starts MLflow UI (optional)

set -e  # Exit on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

echo "========================================================================"
echo "HPP House Price Prediction - Complete Pipeline Runner"
echo "========================================================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please create it first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "[1] Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Run training pipeline
echo "========================================================================"
echo "[2] RUNNING TRAINING PIPELINE"
echo "========================================================================"
python "$PROJECT_ROOT/run_pipeline.py" || {
    echo "❌ Training pipeline failed"
    exit 1
}
echo ""

# Run deployment service
echo "========================================================================"
echo "[3] RUNNING DEPLOYMENT SERVICE"
echo "========================================================================"
python "$PROJECT_ROOT/run_deployment.py" || {
    echo "❌ Deployment service failed"
    exit 1
}
echo ""

# Run sample predictions
echo "========================================================================"
echo "[4] RUNNING SAMPLE PREDICTIONS"
echo "========================================================================"
python "$PROJECT_ROOT/sample_predict.py" || {
    echo "❌ Sample predictions failed"
    exit 1
}
echo ""

echo "========================================================================"
echo "✓ COMPLETE PIPELINE EXECUTION SUCCESSFUL"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. View training metrics:"
echo "     mlflow ui"
echo "     Then open http://localhost:5000"
echo ""
echo "  2. View project structure:"
echo "     see README.md for details"
echo ""
