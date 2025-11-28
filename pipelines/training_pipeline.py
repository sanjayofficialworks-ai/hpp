from zenml import Model, pipeline
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_misssing_values_step import handle_missing_values_step
from steps.outlier_detection_step import outlier_detection_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step

@pipeline(
    model=Model(
        name="price_predictor"
    )
)
def ml_pipeline(file_path: str):
    """
    ZenML pipeline for training a price predictor model.

    This pipeline ingests data, preprocesses it, trains a model, and evaluates it.

    Parameters:
    file_path (str): Path to the data file.
    """
    raw_data = data_ingestion_step(file_path=file_path)
    
    # Handle missing values
    cleaned_data = handle_missing_values_step(df=raw_data, strategy="mean")

    # Detect and handle outliers
    # Note: The outlier detection step seems to have a bug, as it takes a column name but then processes all numeric columns.
    # For now, we will pass 'SalePrice' as the column name.
    outlier_free_data = outlier_detection_step(df=cleaned_data, column_name="SalePrice")

    # Feature engineering
    # Note: The feature engineering step is very basic. We will use log transformation for now.
    featured_data = feature_engineering_step(df=outlier_free_data, strategy="log", features=["SalePrice"])

    # Split the data
    X_train, X_test, y_train, y_test = data_splitter_step(df=featured_data, target_column="SalePrice")

    # Build and train the model
    trained_model = model_building_step(X_train=X_train, y_train=y_train)

    # Evaluate the model
    evaluation_metrics, mse = model_evaluator_step(trained_model=trained_model, X_test=X_test, y_test=y_test)
    
    return evaluation_metrics, mse