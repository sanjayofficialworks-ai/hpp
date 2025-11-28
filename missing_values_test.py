# Import necessary libraries
import pandas as pd
import numpy as np
from analyze_src.missing_values_analysis import MissingValuesAnalyzer, SimpleMissingValueAnalysis

# Set pandas display options for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load the dataset
data_path = "extracted_data/AmesHousing.csv"
df = pd.read_csv(data_path)

# Step 1: Missing values analysis
# Initialize the missing values analyzer with a strategy for simple analysis
missing_analyzer = MissingValuesAnalyzer(SimpleMissingValueAnalysis())
missing_analyzer.analyze(df)
