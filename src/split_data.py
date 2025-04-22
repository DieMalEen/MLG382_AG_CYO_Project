# This program is used to clean and split Excel data into training and testing sets
import pandas as pd
from sklearn.model_selection import train_test_split

# Input the excel file name that needs to be split into sets
filename = "Syntoms.csv"

# Load the dataset into seperate columns
df = pd.read_csv(f"data/{filename}", delimiter=",")

# Remove rows with missing values
df.dropna(inplace=True)

# Remove classes with only one sample
df = df.groupby("prognosis").filter(lambda x: len(x) > 1)

# Split into train and test sets based on 20% test size
# Data is split so GradeClass is proportional in each set
train_data, test_data = train_test_split(df, test_size=0.2, random_state=10, stratify=df["prognosis"])

# Save to CSV files
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)

train_data = pd.read_csv("data/train_data.csv", delimiter=",")

print("prognosis Distribution (Counts):")
print(train_data["prognosis"].value_counts().sort_index())