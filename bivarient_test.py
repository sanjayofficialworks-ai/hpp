# Import necessary libraries
import pandas as pd
import numpy as np
from analyze_src.bivarient_analysis import BivariateAnalyzer, NumericalVsNumericalAnalysis, CategoricalVsNumericalAnalysis

# Set pandas display options for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load the dataset
data_path = "extracted_data/AmesHousing.csv"
df = pd.read_csv(data_path)

# Step 1: Bivariate analysis
# Analyze numerical vs numerical features
numerical_analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
numerical_analyzer.execute_analysis(df, "Gr Liv Area", "SalePrice")

# Analyze categorical vs numerical features
categorical_analyzer = BivariateAnalyzer(CategoricalVsNumericalAnalysis())
categorical_analyzer.execute_analysis(df, "Overall Qual", "SalePrice")
