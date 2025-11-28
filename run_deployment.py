import click
from rich import print
import json
import numpy as np
import pandas as pd
from zenml_backup.integrations.mlflow.services import MLFlowDeploymentService


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """Run the deployment and inference workflow"""
    model_name = "price_predictor"

    if stop_service:
        print("Service stop requested (no active server to stop in local mode)")
        return

    print("="*80)
    print("[1] CREATING DEPLOYMENT SERVICE")
    print("="*80)
    
    # Create and register the local service
    service = MLFlowDeploymentService(
        name="local_mlflow_service",
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name=model_name,
    )
    print(f"✓ Service created: {service}")
    print(f"✓ Prediction endpoint: {service.prediction_url}")
    
    service.start(timeout=10)
    print("✓ Service started")

    print("\n" + "="*80)
    print("[2] RUNNING INFERENCE ON SAMPLE DATA")
    print("="*80)
    
    # Sample test data
    test_data = {
        "data": [
            [1, 526298200, 20, 141.0, 31770, 6, 5, 1960, 1961, 0.0, 416.0, 0.0, 
             1158.0, 1574.0, 1574.0, 0.0, 0.0, 1574.0, 1.0, 0.0, 1.0, 0.0, 3.0, 
             1.0, 5.0, 2.0, 1961.0, 2.0, 528.0, 0.0, 62.0, 0.0, 0.0, 0.0, 0.0, 
             0.0, 5.0, 2010.0],
            [2, 526298205, 20, 80.0, 11622, 5, 6, 1961, 1961, 0.0, 0.0, 0.0, 
             882.0, 882.0, 1096.0, 0.0, 0.0, 1096.0, 0.0, 0.0, 1.0, 1.0, 3.0, 
             1.0, 5.0, 0.0, 1961.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
             0.0, 4.0, 2010.0]
        ]
    }
    
    expected_columns = [
        "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area", "Overall Qual", 
        "Overall Cond", "Year Built", "Year Remod/Add", "Mas Vnr Area", 
        "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", 
        "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Bsmt Full Bath", 
        "Bsmt Half Bath", "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", 
        "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area", 
        "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", 
        "Pool Area", "Misc Val", "Mo Sold", "Yr Sold"
    ]
    
    df = pd.DataFrame(test_data["data"], columns=expected_columns)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)
    
    print(f"✓ Loaded {len(df)} test samples")
    print(f"✓ Data shape: {data_array.shape}")
    
    # Make predictions
    prediction = service.predict(data_array)
    print(f"✓ Predictions: {prediction}")

    print("\n" + "="*80)
    print("DEPLOYMENT AND INFERENCE COMPLETED")
    print("="*80)
    print(f"✓ Successfully made {len(prediction)} predictions")
    print(f"✓ Model is ready for inference")
    print(f"\nTo view training metrics, run:")
    print(f"  mlflow ui")
    print(f"\nThe MLflow UI will be available at http://localhost:5000")


if __name__ == "__main__":
    run_main()
