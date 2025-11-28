# Import necessary libraries
import pandas as pd
import numpy as np
from analyze_src.univarient_analysis import NumericalUnivariateAnalysis, CategoricalUnivariateAnalysis

# Set pandas display options for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load the dataset
data_path = "extracted_data/AmesHousing.csv"
df = pd.read_csv(data_path)

# Step 1: Univariate analysis
# Analyze numerical feature
numerical_analyzer = NumericalUnivariateAnalysis()
numerical_analyzer.analyze(df, 'SalePrice')

# Analyze categorical feature
categorical_analyzer = CategoricalUnivariateAnalysis()
categorical_analyzer.analyze(df, 'Neighborhood')
