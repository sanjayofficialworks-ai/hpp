"""Stub for MLFlow model deployer integration used by the project.

This is a minimal implementation to satisfy imports and allow the
deployment-related scripts to run in a development environment.
"""
from typing import List
from .services import MLFlowDeploymentService


class MLFlowModelDeployer:
    """Minimal model deployer with a simple registry of services."""

    _active_deployer = None

    @classmethod
    def get_active_model_deployer(cls):
        if cls._active_deployer is None:
            cls._active_deployer = MLFlowModelDeployer()
        return cls._active_deployer

    def __init__(self):
        # Keep a small in-memory registry of deployed services
        self._services = []

    def register_service(self, service: MLFlowDeploymentService):
        self._services.append(service)

    def find_model_server(self, pipeline_name: str, pipeline_step_name: str, model_name: str = None) -> List[MLFlowDeploymentService]:
        # Return any services matching pipeline and step name
        matches = [s for s in self._services if s.pipeline_name == pipeline_name and s.pipeline_step_name == pipeline_step_name]
        if model_name:
            matches = [s for s in matches if s.model_name == model_name]
        return matches
