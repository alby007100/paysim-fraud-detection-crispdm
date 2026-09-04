import pandas as pd
from pathlib import Path

# Define project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Fraud_Detection_Dataset.csv"

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Display the first five observations
print("\nFirst five rows:")
print(data.head())

# Display dataset structure and data types
print("\nDataset Information:")
data.info()

# Display dataset dimensions
print("\nDataset Shape:")
print(data.shape)

# Display target class distribution
print("\nFraud Class Distribution:")
print(data["isFraud"].value_counts())

print("\nFraud Percentage:")
print(data["isFraud"].value_counts(normalize=True) * 100)
