from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MissingValuesAnalysisStrategy(ABC):
    """
    Abstract base class for missing values analysis strategies.
    """
    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identify missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.

        Returns:
        None: Prints the missing values information.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualize missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to visualize.

        Returns:
        None: Displays the visualization.
        """
        pass

class SimpleMissingValueAnalysis(MissingValuesAnalysisStrategy):
    """
    Strategy for simple missing values analysis using heatmap.
    """
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identify and print missing values by column.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        """
        print("\nMissing values by column:")
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        if missing_columns.empty:
            print("No missing values found.")
        else:
            print(missing_columns)

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualize missing values using a heatmap.

        Parameters:
        df (pd.DataFrame): The dataframe to visualize.
        """
        print("\nVisualizing Missing Values...")
        try:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
            plt.title("Missing Values Heatmap")
            plt.savefig("missing_values_heatmap.png")
            plt.close()
            print("Plot saved as 'missing_values_heatmap.png'")
        except Exception as e:
            print(f"Skipping heatmap visualization due to error: {e}")

class MissingValuesAnalyzer:
    """
    Class to execute missing values analysis using different strategies.
    """
    def __init__(self, strategy: MissingValuesAnalysisStrategy):
        """
        Initialize the analyzer with a strategy.

        Parameters:
        strategy (MissingValuesAnalysisStrategy): The analysis strategy to use.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: MissingValuesAnalysisStrategy):
        """
        Set a new analysis strategy.

        Parameters:
        strategy (MissingValuesAnalysisStrategy): The new strategy to use.
        """
        self.strategy = strategy

    def analyze(self, df: pd.DataFrame):
        """
        Execute the analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        """
        self.strategy.identify_missing_values(df)
        self.strategy.visualize_missing_values(df)

if __name__ == "__main__":
    df = pd.read_csv("extracted_data/AmesHousing.csv")
    analyzer = MissingValuesAnalyzer(strategy=SimpleMissingValueAnalysis())
    analyzer.analyze(df)
