import pandas as pd
import numpy as np
import pickle

# Load log models and scaler
with open("artifacts/logistic_regression_models.pkl", "rb") as f:
    log_models = pickle.load(f)


with open("artifacts/logistic_regression_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load ran models
with open("artifacts/random_forest_models.pkl", "rb") as f:
    ran_models = pickle.load(f)

# Load XGB
with open("artifacts/xgboost_model_RainTomorrow.pkl", "rb") as f:
    xgboost_model_rain = pickle.load(f)

with open("artifacts/xgboost_model_TempCategory.pkl", "rb") as f:
    xgboost_model_temp = pickle.load(f)

# Mapping for predictions
rain_map = {0: "no", 1: "yes"}
temp_map = {0: "very cold", 1: "cold", 2: "medium", 3: "warm", 4: "very warm"}

# Input string
input_string = "2015-07-28,Watsonia,7.3,14.6,0.0,1.8,8.1,W,24.0,WSW,SSW,7.0,11.0,91.0,68.0,1035.3,1033.3,2.0,6.0,9.3,12.5,No,No,12.3,0.6,59.0,1.2000000000000035,10.833333333333343,7,209,-0.4405187843504952,-0.8977433935342336,850.0,10.95,Very Cold"

# Define columns
columns = ["Date", "Location", "MinTemp", "MaxTemp","Rainfall", "Evaporation", "Sunshine", 
           "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", 
           "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", 
           "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow", 
           "Temp3pm_lag1", "Rainfall_lag1", "Humidity3pm_lag1", "Rainfall_roll3", 
           "Temp3pm_roll3", "Month", "DayOfYear", "Day_sin", "Day_cos", 
           "TempHumidInteraction", "DayAvgTemp", "TempCategory"]

# Parse input
values = input_string.split(",")
df_input = pd.DataFrame([values], columns=columns)

# Drop target labels (not features)
df_input = df_input.drop(columns=["TempCategory", "RainTomorrow", "DayAvgTemp", "MinTemp", "MaxTemp", "Temp3pm", "Temp9am"])

# Encode categoricals
for col in df_input.select_dtypes(include="object").columns:
    df_input[col] = df_input[col].astype("category").cat.codes

# Convert to correct numeric type
df_input = df_input.apply(pd.to_numeric, errors='coerce')

# Scale
x_scaled = scaler.transform(df_input)

# Predict RainTomorrow
log_pred_rain = log_models["RainTomorrow"].predict(x_scaled)[0]
ran_pred_rain = ran_models["RainTomorrow"].predict(x_scaled)[0]
xgb_pred_rain = xgboost_model_rain.predict(x_scaled)[0]

# Predict TempCategory
log_pred_temp = log_models["TempCategory"].predict(x_scaled)[0]
ran_pred_temp = ran_models["TempCategory"].predict(x_scaled)[0]
xgb_pred_temp = xgboost_model_temp.predict(x_scaled)[0]

print("RainTomorrow Predictions")
print("Logistic Regression:", rain_map[log_pred_rain])
print("Random Forest:", rain_map[ran_pred_rain])
print("XGBoost:", rain_map[xgb_pred_rain])

print("\nTempCategory Predictions")
print("Logistic Regression:", temp_map[log_pred_temp])
print("Random Forest:", temp_map[ran_pred_temp])
print("XGBoost:", temp_map[xgb_pred_temp])