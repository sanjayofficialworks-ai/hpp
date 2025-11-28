import os

from pipelines.training_pipeline import ml_pipeline
from zenml import Model
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml_backup.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml_backup.integrations.mlflow.services import MLFlowDeploymentService

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline (we don't depend on its return value here)
    ml_pipeline(file_path="/home/sanjaylinux/hpp/data/AmesHousing.csv")

    # (Re)deploy the trained model by registering a local MLflow deployment service
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    service = MLFlowDeploymentService(
        name="local_mlflow_service",
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="price_predictor",
    )
    deployer.register_service(service)


@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    # Load batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run predictions on the batch data
    predictor(service=model_deployment_service, input_data=batch_data)
