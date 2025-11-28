import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    ZenML step for data ingestion.

    This step ingests data from a ZIP file containing CSV data using the DataIngestorFactory.

    Parameters:
    file_path (str): Path to the ZIP file to ingest data from.

    Returns:
    pd.DataFrame: The ingested dataframe.
    """
    file_extension = ".zip"
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)
    return df
