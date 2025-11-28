from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class BivariateAnalysisStrategy(ABC):
    """
    Abstract base class for bivariate analysis strategies.
    """
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The first feature name.
        feature2 (str): The second feature name.
        """
        pass

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """
    Strategy for numerical vs numerical bivariate analysis using scatter plot.
    """
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Analyze two numerical features with scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The first numerical feature.
        feature2 (str): The second numerical feature.
        """
        plt.figure(figsize=(10, 6))
        try:
            sns.scatterplot(x=feature1, y=feature2, data=df)
            plt.title(f"{feature1} vs {feature2}")
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.savefig(f"{feature1}_vs_{feature2}_scatter.png")
            plt.close()
            print(f"Scatter plot saved as '{feature1}_vs_{feature2}_scatter.png'")
        except Exception as e:
            print(f"Skipping scatterplot for {feature1} vs {feature2} due to plotting error: {e}")

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """
    Strategy for categorical vs numerical bivariate analysis using box plot.
    """
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Analyze categorical vs numerical features with box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The categorical feature.
        feature2 (str): The numerical feature.
        """
        plt.figure(figsize=(10, 6))
        try:
            sns.boxplot(x=feature1, y=feature2, data=df)
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.xticks(rotation=45)
            plt.savefig(f"{feature1}_vs_{feature2}_boxplot.png")
            plt.close()
            print(f"Box plot saved as '{feature1}_vs_{feature2}_boxplot.png'")
        except Exception as e:
            print(f"Skipping boxplot for {feature1} vs {feature2} due to plotting error: {e}")

class BivariateAnalyzer:
    """
    Class to execute bivariate analysis using different strategies.
    """
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initialize the analyzer with a strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The analysis strategy to use.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Set a new analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to use.
        """
        self.strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Execute the analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        feature1 (str): The first feature.
        feature2 (str): The second feature.
        """
        self.strategy.analyze(df, feature1, feature2)

if __name__ == "__main__":
    df = pd.read_csv("extracted_data/AmesHousing.csv")
    analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
    analyzer.execute_analysis(df, "Gr Liv Area", "SalePrice")
    analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    analyzer.execute_analysis(df, "Overall Qual", "SalePrice")
