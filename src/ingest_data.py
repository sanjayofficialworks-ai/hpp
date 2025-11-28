import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

class DataIngestor(ABC):
    """
    Abstract base class for data ingestion strategies.
    """
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest data from a file and return a pandas DataFrame.

        Parameters:
        file_path (str): Path to the file to ingest.

        Returns:
        pd.DataFrame: The ingested data.
        """
        pass

class ZipDataIngestor(DataIngestor):
    """
    Strategy for ingesting data from ZIP files containing CSV.
    """
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest data from a ZIP file, extract CSV, and return DataFrame.

        Parameters:
        file_path (str): Path to the ZIP file.

        Returns:
        pd.DataFrame: The ingested data.

        Raises:
        ValueError: If the file is not a ZIP file or multiple CSV files are found.
        FileNotFoundError: If no CSV file is found in the ZIP.
        """
        # Support both ZIP archives and direct CSV paths.
        if file_path.endswith(".zip"):
            # Extract from the zipfile
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall("extracted_data")
            # Find the extracted CSV file (assuming there is one CSV)
            extracted_files = os.listdir("extracted_data")
            csv_files = [f for f in extracted_files if f.endswith(".csv")]
            if len(csv_files) == 0:
                raise FileNotFoundError("No CSV file found in the ZIP archive")
            if len(csv_files) > 1:
                raise ValueError("More than one CSV file found in the ZIP archive")
            csv_file_path = os.path.join("extracted_data", csv_files[0])
        elif file_path.endswith(".csv"):
            csv_file_path = file_path
        else:
            raise ValueError("Unsupported file type for ingestion; expected .zip or .csv")

        df = pd.read_csv(csv_file_path)
        return df

class DataIngestorFactory:
    """
    Factory class to create appropriate data ingestors based on file extension.
    """
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Get the appropriate data ingestor for the file extension.

        Parameters:
        file_extension (str): The file extension (e.g., '.zip').

        Returns:
        DataIngestor: The corresponding ingestor instance.

        Raises:
        ValueError: If no ingestor is available for the extension.
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")

if __name__ == "__main__":
    file_path = "data/archive.zip"
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension=".zip")
    df = data_ingestor.ingest(file_path=file_path)
    print(df.head())
