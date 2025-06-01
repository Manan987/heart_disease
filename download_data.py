import pandas as pd
import urllib.request

# Download the heart disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data = pd.read_csv(url, header=None)

# Assign column names
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
data.columns = column_names

# Clean the data
data = data.replace('?', pd.NA)
data = data.dropna()

# Convert all columns to numeric
for column in data.columns:
    data[column] = pd.to_numeric(data[column])

# Save the cleaned dataset
data.to_csv('heart_disease_data.csv', index=False)
print("Dataset downloaded and saved as 'heart_disease_data.csv'")