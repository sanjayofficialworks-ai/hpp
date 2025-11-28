# Import necessary libraries
import pandas as pd
import numpy as np
from analyze_src.basic_data_inspection import DataInspector, DataTypesInspectionStrategy, SummaryInspectionStrategy

# Set pandas display options for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Load the dataset
data_path = "extracted_data/AmesHousing.csv"
df = pd.read_csv(data_path)

# Step 1: Basic data inspection
# Initialize the data inspector with a strategy for data type inspection
data_inspection = DataInspector(DataTypesInspectionStrategy())
data_inspection.execute_inspection(df)

# Switch to summary inspection strategy
data_inspection.set_strategy(SummaryInspectionStrategy())
data_inspection.execute_inspection(df)
