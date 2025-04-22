# This program is used to clean and split Excel data into training and testing sets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Input the excel file name that needs to be split into sets
filename = "weatherAUS.csv"

# Load the dataset into separate columns
df = pd.read_csv(f"data/{filename}", delimiter=",")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Location and Date for time series processing
df.sort_values(by=['Location', 'Date'], inplace=True)

# ====================
# Feature Engineering
# ====================

# Lag features (previous day values)
df['Temp3pm_lag1'] = df.groupby('Location')['Temp3pm'].shift(1)
df['Rainfall_lag1'] = df.groupby('Location')['Rainfall'].shift(1)
df['Humidity3pm_lag1'] = df.groupby('Location')['Humidity3pm'].shift(1)

# Rolling averages (past 3 days)
df['Rainfall_roll3'] = df.groupby('Location')['Rainfall'].rolling(3).mean().reset_index(0, drop=True)
df['Temp3pm_roll3'] = df.groupby('Location')['Temp3pm'].rolling(3).mean().reset_index(0, drop=True)

# Date features
df['Month'] = df['Date'].dt.month
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Day_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['Day_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# Interaction feature
df['TempHumidInteraction'] = df['Temp3pm'] * df['Humidity3pm']

# Drop rows with missing values caused by lag/rolling features
df.dropna(inplace=True)

# Split into train and test sets based on 20% test size
# Data is split so RainTomorrow is proportional in each set
train_data, test_data = train_test_split(df, test_size=0.2, random_state=10, stratify=df["RainTomorrow"])

# Save to CSV files
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)

# Optional: Load and show class distribution
train_data = pd.read_csv("data/train_data.csv", delimiter=",")
print("RainTomorrow Distribution (Counts):")
print(train_data["RainTomorrow"].value_counts().sort_index())