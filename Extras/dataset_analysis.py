import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# TENNIS


# # Load the dataset with low_memory set to False to handle mixed types better
# data = pd.read_csv('tennis_data.csv', low_memory=False)

# # Convert columns that should be numeric but are loaded as object type
# # This tries to convert all columns to numeric and those that cannot be converted will be coerced to NaN
# for col in data.columns:
#     data[col] = pd.to_numeric(data[col], errors='coerce')

# # Drop all columns that are completely non-numeric or have been coerced to NaN (if any)
# numeric_data = data.select_dtypes(include=[np.number])

# # Compute the correlation matrix
# corr_matrix = numeric_data.corr()

# # Create a heatmap of the correlation matrix
# plt.figure(figsize=(12, 10))  # You might need to adjust the figure size depending on the number of columns
# sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

# # Display the number of rows and columns in the dataset
# print(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

# # Count and display the number of missing values in each column
# missing_values = data.isnull().sum()
# print("Number of missing values in each column:")
# print(missing_values)

# # Display the number of missing values in the numeric columns as well
# print("Number of missing values in numeric columns:")
# print(numeric_data.isnull().sum())

# # Display basic statistical summaries of the numeric data
# print("Statistical summary of numeric columns:")
# print(numeric_data.describe())

# # Additional information that might be useful
# # Check for the presence of duplicate rows
# duplicates = data.duplicated().sum()
# print(f"Number of duplicate rows: {duplicates}")

# # Distribution of missing values
# percent_missing = (missing_values / data.shape[0]) * 100
# print("Percentage of missing values per column:")
# print(percent_missing)


# PREMIER LEAGUE



# Load the dataset with low_memory set to False to handle mixed types better
data = pd.read_csv('PremierLeague.csv', low_memory=False)

# Convert season to the starting year as an integer for comparison
data['Season_Start_Year'] = data['Season'].str.split('-').str[0].astype(int)

# Filter out seasons from 1999 and before
data = data[data['Season_Start_Year'] > 1999]

# Drop the temporary 'Season_Start_Year' column if no longer needed
data.drop('Season_Start_Year', axis=1, inplace=True)


# Convert columns that should be numeric but are loaded as object type
# This tries to convert all columns to numeric and those that cannot be converted will be coerced to NaN
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop all columns that are completely non-numeric or have been coerced to NaN (if any)
numeric_data = data.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr_matrix = numeric_data.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))  # You might need to adjust the figure size depending on the number of columns
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Display the number of rows and columns in the dataset
print(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

# Count and display the number of missing values in each column
missing_values = data.isnull().sum()
print("Number of missing values in each column:")
print(missing_values)

# Display the number of missing values in the numeric columns as well
print("Number of missing values in numeric columns:")
print(numeric_data.isnull().sum())

# Display basic statistical summaries of the numeric data
print("Statistical summary of numeric columns:")
print(numeric_data.describe())

# Additional information that might be useful
# Check for the presence of duplicate rows
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Distribution of missing values
percent_missing = (missing_values / data.shape[0]) * 100
print("Percentage of missing values per column:")
print(percent_missing)