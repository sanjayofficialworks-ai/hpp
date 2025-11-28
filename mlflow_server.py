#!/usr/bin/env python
"""
Simple MLflow server script that loads and serves the trained model.
"""

import os
import subprocess
import sys

# Path to MLflow runs directory (from the ZenML stack)
mlflow_backend_uri = "file:///home/sanjaylinux/.config/zenml/local_stores/7bb690eb-01f9-498f-9546-2aec4ac85231/mlruns"

# Extract the path from the file URI
runs_dir = mlflow_backend_uri.replace("file://", "")

print(f"Starting MLflow UI server with backend-store-uri: {mlflow_backend_uri}")
print(f"MLflow runs directory: {runs_dir}")
print(f"Open browser: http://localhost:5000")
print()

# Start MLflow UI
try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            mlflow_backend_uri,
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ],
        check=False,
    )
except KeyboardInterrupt:
    print("\nMLflow server stopped.")
