import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying the same preprocessing to the test data.")

    # Apply the preprocessing and model prediction
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # Initialize the evaluator with the regression strategy
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

    # Perform the evaluation
    evaluation_metrics = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )

    # Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse = evaluation_metrics.get("Mean Squared Error", None)
    r_squared = evaluation_metrics.get("R-Squared", None)
    
    # Log metrics to MLflow
    try:
        mlflow.log_metric("Mean Squared Error", float(mse))
        mlflow.log_metric("R-Squared", float(r_squared))
    except Exception as e:
        logging.warning(f"Failed to log metrics to MLflow: {e}")
    
    logging.info(f"[model_evaluator_step] Model Evaluation Metrics: {evaluation_metrics}")
    
    return evaluation_metrics, mse
