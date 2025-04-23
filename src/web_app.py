import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import pickle
import torch
import numpy as np
import dash_bootstrap_components as dbc
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import os

# Load artifacts
with open("artifacts/logistic_regression_models.pkl", "rb") as f:
    logreg_models = pickle.load(f)
with open("artifacts/logistic_regression_scaler.pkl", "rb") as f:
    logreg_scaler = pickle.load(f)
with open("artifacts/rain_encoder.pkl", "rb") as f:
    rain_encoder = pickle.load(f)
with open("artifacts/temp_encoder.pkl", "rb") as f:
    temp_encoder = pickle.load(f)

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

    dbc.Row([
        dbc.Col([
            html.Label("Date"),
            dcc.DatePickerSingle(
                id="input-Date",
                placeholder="Select a date",
                display_format="YYYY-MM-DD",
                className="form-control"
            )
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label(f"{feature}"),
            dcc.Dropdown(
                id=f"input-{feature}",
                options=[{"label": v, "value": v} for v in categorical_inputs[feature]],
                className="form-control"
            )
        ]) if feature in categorical_inputs else dbc.Col([
            html.Label(f"{feature}"),
            dcc.Input(type="number", id=f"input-{feature}", placeholder=f"Enter {feature}", className="form-control")
        ])
        for feature in user_input_features if feature != "Date"
    ], className="mb-4", style={"columnCount": 2}),

    dbc.Button("Predict", id="predict-button", color="primary", className="mb-3 w-100"),

    html.Div(id="prediction-output", className="text-center h4 text-success")
])


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
    
    # Predict
    rain_pred = logreg_models["RainTomorrow"].predict(df_scaled)[0]
    temp_pred = logreg_models["TempCategory"].predict(df_scaled)[0]

    return f"Rain Tomorrow: {rain_encoder.inverse_transform([rain_pred])[0]} | Temperature: {temp_encoder.inverse_transform([temp_pred])[0]}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
