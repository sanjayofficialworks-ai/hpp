#!/usr/bin/env python
"""
Complete end-to-end ML pipeline with predictions.
This script runs the full pipeline and demonstrates inference.
"""

import os
import sys
import subprocess
import time

PROJECT_DIR = "/home/sanjaylinux/hpp"
VENV_BIN = os.path.join(PROJECT_DIR, ".venv", "bin")
MLFLOW_BACKEND_URI = "file:///home/sanjaylinux/.config/zenml/local_stores/7bb690eb-01f9-498f-9546-2aec4ac85231/mlruns"

print("=" * 80)
print("ML PIPELINE - END-TO-END WITH PREDICTIONS")
print("=" * 80)

# Step 1: Run training pipeline
print("\n[STEP 1] Running Training Pipeline...")
print("-" * 80)
result = subprocess.run(
    [os.path.join(VENV_BIN, "python"), "run_pipeline.py"],
    cwd=PROJECT_DIR,
    capture_output=False
)
if result.returncode != 0:
    print("✗ Training pipeline failed")
    sys.exit(1)
print("✓ Training pipeline completed successfully")

# Step 2: Run deployment pipeline
print("\n[STEP 2] Running Deployment & Inference Pipeline...")
print("-" * 80)
result = subprocess.run(
    [os.path.join(VENV_BIN, "python"), "run_deployment.py"],
    cwd=PROJECT_DIR,
    capture_output=False
)
if result.returncode != 0:
    print("✗ Deployment pipeline failed")
    sys.exit(1)
print("✓ Deployment pipeline completed successfully")

# Step 3: Start MLflow server
print("\n[STEP 3] Starting MLflow UI Server...")
print("-" * 80)
mlflow_log_file = "/tmp/mlflow_server.log"
mlflow_process = subprocess.Popen(
    [
        os.path.join(VENV_BIN, "python"),
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        MLFLOW_BACKEND_URI,
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
    ],
    cwd=PROJECT_DIR,
    stdout=open(mlflow_log_file, "w"),
    stderr=subprocess.STDOUT,
)
print(f"✓ MLflow server started (PID: {mlflow_process.pid})")
print(f"  Dashboard: http://localhost:5000")
time.sleep(4)

# Step 4: Run sample prediction
print("\n[STEP 4] Running Sample Prediction...")
print("-" * 80)
result = subprocess.run(
    [os.path.join(VENV_BIN, "python"), "sample_predict.py"],
    cwd=PROJECT_DIR,
    capture_output=False
)
print("\n✓ Sample prediction script completed")

# Step 5: Summary
print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print("\nResults Summary:")
print(f"✓ Training: Model trained and evaluated")
print(f"✓ Deployment: Inference pipeline executed")
print(f"✓ Prediction: Sample data processed")
print(f"\nMLflow Dashboard: http://localhost:5000")
print(f"Server PID: {mlflow_process.pid}")
print("\nTo view training metrics, open browser to: http://localhost:5000")
print("To stop the server, run: kill {mlflow_process.pid}")
print("\nPress Ctrl+C to stop the server...")
print("=" * 80)

# Keep server running
try:
    mlflow_process.wait()
except KeyboardInterrupt:
    print("\n\nShutting down MLflow server...")
    mlflow_process.terminate()
    mlflow_process.wait()
    print("✓ Server stopped")
