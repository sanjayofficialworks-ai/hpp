"""Stub definitions for MLflow deployment service used by the project."""

class MLFlowDeploymentService:
    """A minimal representation of a deployed model service.

    Provides `start`, `stop`, `predict` and a `prediction_url` so the
    project's predictor and deployment scripts can interact with it.
    """

    def __init__(self, name: str, pipeline_name: str, pipeline_step_name: str, model_name: str):
        self.name = name
        self.pipeline_name = pipeline_name
        self.pipeline_step_name = pipeline_step_name
        self.model_name = model_name
        self._running = False
        self.prediction_url = f"http://localhost:5000/{self.name}"

    def start(self, timeout: int = 10):
        # Simulate starting a service; set running flag and return
        self._running = True

    def stop(self, timeout: int = 10):
        # Simulate stopping a service
        self._running = False

    def predict(self, data):
        # Return a zero array compatible with the caller's expected shape
        try:
            import numpy as _np

            # If data is 2D, return zeros of shape (n_rows,)
            if hasattr(data, "shape"):
                n = data.shape[0]
                return _np.zeros(n)
        except Exception:
            pass

        return []

    def __repr__(self):
        return f"MLFlowDeploymentService(name={self.name}, pipeline={self.pipeline_name}, step={self.pipeline_step_name})"
