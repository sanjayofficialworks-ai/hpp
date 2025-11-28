from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):
    """
    Abstract base class for univariate analysis strategies.
    """
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a given feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The feature name to analyze.
        """
        pass

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """
    Strategy for numerical univariate analysis using histogram.
    """
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Analyze numerical feature with histogram.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The numerical feature name to analyze.
        """
        plt.figure(figsize=(10, 6))
        try:
            sns.histplot(df[feature], kde=True, bins=30)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.savefig(f"{feature}_histogram.png")
            plt.close()
            print(f"Histogram saved as '{feature}_histogram.png'")
        except Exception as e:
            print(f"Skipping histogram for {feature} due to plotting error: {e}")

class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """
    Strategy for categorical univariate analysis using count plot.
    """
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Analyze categorical feature with count plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The categorical feature name to analyze.
        """
        plt.figure(figsize=(10, 6))
        try:
            sns.countplot(x=feature, data=df, palette="muted")
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.savefig(f"{feature}_countplot.png")
            plt.close()
            print(f"Count plot saved as '{feature}_countplot.png'")
        except Exception as e:
            print(f"Skipping countplot for {feature} due to plotting error: {e}")

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("extracted_data/AmesHousing.csv")

    # Analyze numerical feature
    numerical_analyzer = NumericalUnivariateAnalysis()
    numerical_analyzer.analyze(df, 'SalePrice')





    # Analyze categorical feature
    categorical_analyzer = CategoricalUnivariateAnalysis()
    categorical_analyzer.analyze(df, 'Neighborhood')
