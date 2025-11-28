from zenml import step
from zenml_backup.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml_backup.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, step_name: str) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    # get the MLflow model deployer stack component (fallback to in-memory stub)
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
    )

    # If no service is registered, create a local fallback service and register it
    if not existing_services:
        service = MLFlowDeploymentService(
            name="local_mlflow_service",
            pipeline_name=pipeline_name,
            pipeline_step_name=step_name,
            model_name="price_predictor",
        )
        model_deployer.register_service(service)
        return service

    return existing_services[0]
