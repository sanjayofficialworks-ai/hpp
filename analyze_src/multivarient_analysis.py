from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MultivariateAnalysisTemplate(ABC):
    """
    Abstract base class for multivariate analysis templates.
    """
    def analyze(self, df: pd.DataFrame):
        """
        Perform multivariate analysis by generating correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate correlation heatmap.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        """
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    """
    Simple implementation of multivariate analysis using heatmap and pair plot.
    """
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and save correlation heatmap.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidth=0.5)
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()
        print("Correlation heatmap saved as 'correlation_heatmap.png'")

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and save pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.savefig("pairplot.png")
        plt.close()
        print("Pair plot saved as 'pairplot.png'")

if __name__ == "__main__":
    df = pd.read_csv("extracted_data/AmesHousing.csv")
    multivariate_analyzer = SimpleMultivariateAnalysis()
    selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]
    multivariate_analyzer.analyze(selected_features)
