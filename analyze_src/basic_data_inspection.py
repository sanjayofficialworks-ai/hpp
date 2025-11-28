from abc import ABC, abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    """
    Abstract base class for data inspection strategies.
    """
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection has to be performed.

        Returns:
        None: This method prints the inspection result directly.
        """
        pass

class DataTypesInspectionStrategy(DataInspectionStrategy):
    """
    Strategy for inspecting data types and non-null counts.
    """
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts.
        """
        print("\nData Types and Non-Null Counts:")
        print(df.info())

class SummaryInspectionStrategy(DataInspectionStrategy):
    """
    Strategy for inspecting summary statistics.
    """
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the summary statistics.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=['O']))

class DataInspector:
    """
    Class to execute data inspections using different strategies.
    """
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initialize the DataInspector with a strategy.

        Parameters:
        strategy (DataInspectionStrategy): The inspection strategy to use.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Set a new inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to use.
        """
        self.strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Execute the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to inspect.
        """
        self.strategy.inspect(df)

if __name__ == "__main__":
    df = pd.read_csv("extracted_data/AmesHousing.csv")
    inspector = DataInspector(strategy=DataTypesInspectionStrategy())
    inspector.execute_inspection(df)
    inspector.set_strategy(SummaryInspectionStrategy())
    inspector.execute_inspection(df)
