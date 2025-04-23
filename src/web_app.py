import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dash_bootstrap_components as dbc
import numpy as np
import os
from datetime import datetime

# Load artifacts
with open("artifacts/logistic_regression_models.pkl", "rb") as f:
    logreg_models = pickle.load(f)
with open("artifacts/logistic_regression_scaler.pkl", "rb") as f:
    logreg_scaler = pickle.load(f)
with open("artifacts/rain_encoder.pkl", "rb") as f:
    rain_encoder = pickle.load(f)
with open("artifacts/temp_encoder.pkl", "rb") as f:
    temp_encoder = pickle.load(f)
    
with open("artifacts/random_forest_models.pkl", "rb") as f:
    rf_models = pickle.load(f)
with open("artifacts/xgboost_model_RainTomorrow.pkl", "rb") as f:
    xgboost_model_rain = pickle.load(f)
with open("artifacts/xgboost_model_TempCategory.pkl", "rb") as f:
    xgboost_model_temp = pickle.load(f)

# Define the Deep Learning model (Dual Output Model)
class DualOutputNN(nn.Module):
    def __init__(self, input_size, rain_classes, temp_classes):
        super(DualOutputNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.out_rain = nn.Linear(32, rain_classes)
        self.out_temp = nn.Linear(32, temp_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out_rain(x), self.out_temp(x)

# Load the deep learning model
deep_learning_model = DualOutputNN(input_size=28, rain_classes=2, temp_classes=5)  # Example sizes, adjust input_size
deep_learning_model.load_state_dict(torch.load("artifacts/weather_deep_model.pth"))
deep_learning_model.eval()  # Set model to evaluation mode

# Define input features (excluding engineered features)
user_input_features = [
    "Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir",
    "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "RainToday", "TempHumidInteraction", "Rainfall_lag1", "Temp3pm_lag1", "Humidity3pm_lag1",
    "Rainfall_roll3", "Temp3pm_roll3"
]

# All features including engineered ones (needed for model input)
final_model_features = user_input_features + ["Month", "DayOfYear", "Day_sin", "Day_cos"]

categorical_inputs = {
    "Location": ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth"],
    "WindGustDir": ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "ENE", "ESE", "WNW", "WSW", "SSE", "SSW", "NNW", "NNE"],
    "WindDir9am": ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "ENE", "ESE", "WNW", "WSW", "SSE", "SSW", "NNW", "NNE"],
    "WindDir3pm": ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "ENE", "ESE", "WNW", "WSW", "SSE", "SSW", "NNW", "NNE"],
    "RainToday": ["No", "Yes"]
}

# App setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H2("Weather Prediction App", className="text-center my-4"),

    # Row for Inputs
    dbc.Row([
        dbc.Col([
            html.Label(f"{feature}"),
            dcc.Dropdown(
                options=[{"label": v, "value": v} for v in categorical_inputs[feature]],
                id=f"input-{feature}",
                className="form-control"
            )
        ], width=4) if feature in categorical_inputs else dbc.Col([
            html.Label(f"{feature}"),
            dcc.Input(
                type="number", id=f"input-{feature}", placeholder=f"Enter {feature}",
                className="form-control"
            )
        ], width=4) for feature in user_input_features if feature != "Date"
    ], className="mb-4"),

    # Row for Date Picker
    dbc.Row([
        dbc.Col([
            html.Label("Date"),
            dcc.DatePickerSingle(
                id="input-Date",
                placeholder="Select a date",
                display_format="YYYY-MM-DD",
                className="form-control"
            )
        ], width=4),
    ], className="mb-4"),

    # Row for the Predict Button
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict", id="predict-button", color="primary", className="w-100"),
        ], width=4),
    ], className="mb-4"),

    # Output Section for Results
    html.Div(
        id="prediction-output", 
        className="text-center h4 text-success my-3"
    ),

], fluid=True)


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [
        State("input-Date", "date")
    ] + [
        State(f"input-{feature}", "value")
        for feature in user_input_features if feature != "Date"
    ],
    prevent_initial_call=True
)
def predict(n_clicks, date, *values):
    if not n_clicks:
        return ""

    input_values = [date] + list(values)
    input_data = dict(zip(user_input_features, input_values))
    
    print("Received inputs:", input_data)

    # Debug: find missing values
    missing = [k for k, v in input_data.items() if v is None]
    print("Missing fields:", missing)

    # Ensure all values are provided
    if any(v is None for v in input_data.values()):
        return "Please fill in all inputs."

    # Process date and add engineered features    
    try:
        date_obj = datetime.strptime(input_data["Date"], "%Y-%m-%d")
    except Exception as e:
        return "Invalid date format."
    
    input_data["Month"] = date_obj.month
    input_data["DayOfYear"] = date_obj.timetuple().tm_yday
    input_data["Day_sin"] = np.sin(2 * np.pi * input_data["DayOfYear"] / 365.0)
    input_data["Day_cos"] = np.cos(2 * np.pi * input_data["DayOfYear"] / 365.0)

    # Create DataFrame with final features
    df = pd.DataFrame([{k: input_data[k] for k in final_model_features}])
    df = df.drop(columns=["MaxTemp", "MinTemp"], errors="ignore")


    # Encode categoricals
    for col in categorical_inputs:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.dayofyear

    # Scale
    with open("artifacts/feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)

    df = df[feature_order]  # Enforce correct order
    print("Expected order:", feature_order)
    print("Current order :", df.columns.tolist())
    df_scaled = logreg_scaler.transform(df)
    
    # Logistic Regression Prediction
    logreg_rain_pred = logreg_models["RainTomorrow"].predict(df_scaled)[0]
    logreg_temp_pred = logreg_models["TempCategory"].predict(df_scaled)[0]

    # Random Forest Prediction
    rf_rain_pred = rf_models["RainTomorrow"].predict(df_scaled)[0]
    rf_temp_pred = rf_models["TempCategory"].predict(df_scaled)[0]

    # XGBoost Prediction
    xgb_rain_pred = xgboost_model_rain.predict(df_scaled)[0]
    xgb_temp_pred = xgboost_model_temp.predict(df_scaled)[0]

    # Deep Learning Prediction (ensure the model is evaluated properly)
    with torch.no_grad():
        deep_input_tensor = torch.tensor(df_scaled, dtype=torch.float32)
        rain_output, temp_output = deep_learning_model(deep_input_tensor)
        deep_rain_pred = rain_output.argmax(dim=1).item()
        deep_temp_pred = temp_output.argmax(dim=1).item()

    # Decode predictions
    rain_predictions = {
        "Logistic Regression": rain_encoder.inverse_transform([logreg_rain_pred])[0],
        "Random Forest Rain": rain_encoder.inverse_transform([rf_rain_pred])[0],
        "XGBoost": rain_encoder.inverse_transform([xgb_rain_pred])[0],
        "Deep Learning": rain_encoder.inverse_transform([deep_rain_pred])[0]
    }

    temp_predictions = {
        "Logistic Regression": temp_encoder.inverse_transform([logreg_temp_pred])[0],
        "Random Forest Temp": temp_encoder.inverse_transform([rf_temp_pred])[0],
        "XGBoost": temp_encoder.inverse_transform([xgb_temp_pred])[0],
        "Deep Learning": temp_encoder.inverse_transform([deep_temp_pred])[0]
    }

    # Display all predictions
    rain_output = " | ".join([f"{model}: {prediction}" for model, prediction in rain_predictions.items()])
    temp_output = " | ".join([f"{model}: {prediction}" for model, prediction in temp_predictions.items()])

    return f"Rain Tomorrow: {rain_output} | Temperature: {temp_output}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
