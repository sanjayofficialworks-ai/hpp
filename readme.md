# House Price Prediction — Project Overview & Output

## Project Objectives
- Build a reproducible end-to-end ML pipeline that ingests the Ames Housing dataset, handles missing values and outliers, performs feature engineering, trains and evaluates a regression model, and provides a deployment/inference flow.
- Demonstrate MLOps best practices: reproducible runs, experiment tracking, model versioning and a simple local deploy/predict flow using ZenML + MLflow.

---

## Quick Start (create environment & install dependencies)
If you're setting this project up locally, follow these three steps to create a virtual environment, activate it, and install the required Python packages. Run these from the project root directory (`/home/sanjaylinux/hpp`).

```bash
python -m venv .venv        # create a local virtual environment named .venv
source .venv/bin/activate   # activate the virtual environment
pip install -r requirements.txt  # install Python dependencies listed in requirements.txt
# House Price Prediction — Project Overview & Output

## Project Objectives
- Build a reproducible end-to-end ML pipeline that ingests the Ames Housing dataset, handles missing values and outliers, performs feature engineering, trains and evaluates a regression model, and provides a deployment/inference flow.
- Demonstrate MLOps best practices: reproducible runs, experiment tracking, model versioning and a simple local deploy/predict flow using ZenML + MLflow.

---

## Quick Start (create environment & install dependencies)
If you're setting this project up locally, follow these three steps to create a virtual environment, activate it, and install the required Python packages. Run these from the project root directory (`/home/sanjaylinux/hpp`).

```bash
python -m venv .venv        # create a local virtual environment named .venv
source .venv/bin/activate   # activate the virtual environment
pip install -r requirements.txt  # install Python dependencies listed in requirements.txt
```

After these commands complete, you're ready to run the pipelines described below.


## Key Output Screenshots (paste the images into the repository next to this README)
- `mlflow_runs_dashboard.png` — MLflow runs and metrics table (R², MSE, training times).
- `pipeline_dag_top.png` — Full ZenML pipeline DAG showing all steps.
- `pipeline_dag_zoom.png` — Zoom into model build / split / evaluator area.
- `metrics_table.png` — Screenshot of metrics (r2, mse, rmse, etc).
- `vscode_tree.png` — Project folder structure in VS Code.
- `prediction_terminal.png` — Terminal showing `PREDICTION COMPLETE` and artifacts path.
- `inference_dag.png` — Deployment/inference DAG (prediction_service_loader → predictor)

> NOTE: If those image files are not present yet, copy the screenshots you saved into the project root and name them as listed above so they render here.

---

## Embedded Screenshots
Below are the screenshots currently available in the `screenshots/` folder. Replace any placeholder images in `screenshots/` with higher-fidelity screenshots if you have them

### Pipeline DAG (top)
Pipeline DAG top
<img width="403" height="572" alt="Screenshot 2025-11-27 212937" src="https://github.com/user-attachments/assets/19cce6e3-4f34-495b-ab36-ec9bef872692" />


### Pipeline DAG (zoom)
<img width="623" height="530" alt="Screenshot 2025-11-27 212953" src="https://github.com/user-attachments/assets/b6dc5a4d-db76-4ba2-8219-6f007038ecbb" />


### Metrics table (placeholder)
<img width="524" height="274" alt="Screenshot 2025-11-27 213053" src="https://github.com/user-attachments/assets/8820e4c1-92d8-49b3-93e8-f4e14cfb38e7" />


### VS Code project tree (placeholder)
<img width="770" height="849" alt="Screenshot 2025-11-27 213313" src="https://github.com/user-attachments/assets/f0b24262-0990-4486-9885-fd1ad74ecd2c" />


### Prediction terminal (placeholder)
<img width="770" height="444" alt="Screenshot 2025-11-27 214748" src="https://github.com/user-attachments/assets/c72de4c9-b04f-4553-acab-b29fadcbb196" />
<img width="857" height="402" alt="Screenshot 2025-11-27 213552" src="https://github.com/user-attachments/assets/b2eb1553-4379-45ab-b5f4-2dc1acaf4a01" />


### Inference DAG (placeholder)
<img width="719" height="568" alt="Screenshot 2025-11-27 214911" src="https://github.com/user-attachments/assets/264f8790-67f2-4488-9a5c-bccdcb200a6d" />


### Additional analysis visuals
SalePrice histogram
<img width="538" height="323" alt="image" src="https://github.com/user-attachments/assets/532b3d66-eec8-4b03-aff3-a556789a7916" />

Correlation heatmap
<img width="765" height="634" alt="image" src="https://github.com/user-attachments/assets/5ccc6c8a-f7b8-4533-a04c-e2009590278d" />

Missing values heatmap
<img width="950" height="632" alt="image" src="https://github.com/user-attachments/assets/6a4bf134-8524-4aea-aa99-b0f0397691c6" />

Neighborhood counts
<img width="1054" height="634" alt="image" src="https://github.com/user-attachments/assets/8211bdad-fc78-4d25-a7fa-457a794f689c" />

Overall Quality vs SalePrice
<img width="1049" height="631" alt="image" src="https://github.com/user-attachments/assets/e6a0fefe-30be-4732-bd05-3ad578d3f926" />

Pairplot sample
<img width="636" height="634" alt="image" src="https://github.com/user-attachments/assets/e5b15177-d5f4-4a40-82d7-91a871e8a2e9" />


---

## Important Textual Outputs (copy of relevant terminal/MLflow info)

### Model evaluation metrics (example values from latest run):
```
training_r2_score: 0.9050842317453247
training_score: 0.9050842317453247
training_mean_squared_error: 0.013211883579276322
training_root_mean_squared_error: 0.11494295793686676
training_mean_absolute_error: 0.08151921401629988
```

### Final pipeline completion message (example):
```
✓ Pipeline completed successfully!
To view experiment metrics, run:
    mlflow ui
Then open http://localhost:5000 in your browser.
```

### Sample prediction output (example):
```
==========================================================
PREDICTION COMPLETE
==========================================================
✓ Successfully made 2 predictions
✓ Model is ready for inference

To run the full inference pipeline:
    python run_deployment.py

[5/5] Displaying model artifacts location...
Model artifacts stored in:
/home/sanjaylinux/.config/zenml/local_stores/7bb690eb-01f9-498f-9546-2aec4ac85231/
```

---

## How to run this project (simple step-by-step)
1. Create & activate virtual environment (if not already created)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Initialize ZenML local repository if you haven't:
```bash
source .venv/bin/activate
zenml init
```

3. Run the training pipeline (produces model + metrics logged to MLflow):
```bash
source .venv/bin/activate
python run_pipeline.py
```

4. Start MLflow UI to inspect runs:
```bash
source .venv/bin/activate
mlflow ui --port 5000
# then open http://localhost:5000
```

5. Run the deployment/inference pipeline (registers local service or stub and runs prediction):
```bash
source .venv/bin/activate
python run_deployment.py
```

6. Make a quick sample prediction (loads latest artifacts):
```bash
python sample_predict.py
```

---

## Simple explanation of the implementation (plain terms)
- Data ingestion: reads `AmesHousing.csv` into a pandas DataFrame.
- Missing values step: fills or imputes missing values (strategy: mean/most_frequent depending on column).
- Outlier detection: uses a simple statistical method (Z-score) to remove extreme values that distort training.
- Feature engineering: performs simple transforms (e.g., log-transform of `SalePrice`) to stabilize variance and help the model.
- Data splitting: creates training and test sets.
- Model training: builds a scikit-learn `Pipeline` that includes preprocessing and `LinearRegression` model.
- Evaluation: computes metrics (MSE, R²) and persists them to MLflow (via autologging and explicit metric writes).
- Deployment: a small local stub or MLflow deployment service is registered; a `predictor` step calls that service to produce predictions on sample data.

---

## Files & Folders — purpose (2 lines each)
- `AmesHousing.csv` — Raw dataset used for training and evaluation (Ames Housing dataset).
- `.venv/` — Python virtual environment containing project dependencies (do not commit).
- `analyze_src/` — EDA and visualization notebooks/scripts (useful for exploration, not required for pipeline runs).
- `data/` — Contains the original dataset and extracted files used by ingestion step.
- `extracted_data/` — Copy of the source dataset prepared for ingestion.
- `mlruns/` — Local MLflow tracking directory (auto-created by MLflow). Contains run artifacts and metrics.
- `pipelines/` — ZenML pipeline definitions (e.g., `training_pipeline.py`, `deployment_pipeline.py`).
- `pandas/` — Local folder (if present) that may shadow the real `pandas` package — **should be removed** before pushing.
- `src/` — Helper utilities and shared modules used by steps (evaluator, helpers, etc.).
- `steps/` — ZenML step implementations (each step is a standalone unit for the pipeline).
- `zenml_backup/` — Local stub implementations for ZenML/MLflow integrations used for local development and testing.
- `run_pipeline.py` — Convenience script to run the training pipeline end-to-end.
- `run_deployment.py` — Script to run the deployment/inference pipeline and start prediction service.
- `sample_predict.py` — Loads the latest trained model artifact and produces example predictions.
- `run.sh` / `run_all.sh` — Bash scripts that orchestrate running training, deployment and sample predictions.
- `requirements.txt` — (If present) lists Python dependencies needed to run the project.
- `read.md` — (This file) Project summary, outputs and run instructions.

---



