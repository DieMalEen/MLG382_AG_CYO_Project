import dash
from dash import html, dcc, Input, Output
import pandas as pd
import pickle
import numpy as np
import os



app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"])
server = app.server

# Load models and scaler
def load_model_and_scaler(model_type):
    if model_type == "deep_learning":
        model = load_model("artifacts/deep_learning_model.keras")
    else:
        model_path = f"artifacts/{model_type}_model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    scaler = None
    if model_type in ["regression", "deep_learning"]:
        with open("artifacts/regression_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

logreg_model, logreg_scaler = load_model_and_scaler("regression")
rf_model, _ = load_model_and_scaler("random_forest")
xgb_model, _ = load_model_and_scaler("xgboost")
dl_model, _ = load_model_and_scaler("deep_learning")

columns = ["Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly", "Absences", 
           "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"]

# Define options for categorical fields
categorical_options = {
    "Gender": [{"label": "Male", "value": 0}, {"label": "Female", "value": 1}],
    "Ethnicity": [{"label": "Group A", "value": 0}, {"label": "Group B", "value": 1}, {"label": "Group C", "value": 2}],
    "ParentalEducation": [{"label": "High School", "value": 0}, {"label": "Diploma", "value": 1}, {"label": "Degree", "value": 2}],
    "Tutoring": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "ParentalSupport": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Extracurricular": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Sports": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Music": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Volunteering": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
}

# Layout of the Dash app
app.layout = html.Div(className="min-h-screen bg-gray-100 flex justify-center items-center p-6", children=[
    html.Div(className="bg-white rounded-xl shadow-lg p-8 max-w-2xl w-full", children=[
        html.H1("BrightPath Academy Grade Prediction", className="text-3xl font-bold text-center text-blue-800 mb-6"),
        
        # Model dropdown
        html.Div(className="mb-6", children=[
            html.Label("Select Model", className="block text-gray-700 font-medium mb-2"),
            dcc.Dropdown(
                id='model-selector',
                options=[
                    {'label': 'Logistic Regression', 'value': 'regression'},
                    {'label': 'Random Forest', 'value': 'random_forest'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'Deep Learning', 'value': 'deep_learning'}
                ],
                value='regression',
                className="border border-gray-300 rounded-lg p-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
            ),
        ]),

        # Input fields
        html.Div(className="mb-6", children=[
            html.Label("Student Details", className="block text-gray-700 font-medium mb-4"),
            html.Div(className="grid grid-cols-1 sm:grid-cols-2 gap-4", children=[
                html.Div([
                    html.Label(f"{col}", className="block text-gray-600 text-sm mb-1"),
                    dcc.Dropdown(
                        id=f'input-{col.lower()}',
                        options=categorical_options[col],
                        value=categorical_options[col][0]['value'],
                        className="border border-gray-300 rounded-lg p-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                    ) if col in categorical_options else
                    dcc.Input(
                        id=f'input-{col.lower()}',
                        type='number',
                        value=0,
                        className="border border-gray-300 rounded-lg p-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                    )
                ]) for col in columns
            ])
        ]),

        # Predict button
        html.Button('Predict Grade', id='predict-button', n_clicks=0, 
                    className="w-full bg-blue-600 text-white font-semibold py-2 rounded-lg hover:bg-blue-700 transition duration-200"),
        
        # Output predicted grade
        html.Div(id='prediction-output', className="mt-6 text-center text-xl font-medium text-blue-800")
    ])
])

# Prediction logic
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('model-selector', 'value'),
    [Input(f'input-{col.lower()}', 'value') for col in columns]
)
def update_prediction(n_clicks, model_type, *input_values):
    if n_clicks == 0:
        return "Enter student details and click Predict Grade."

    input_data = {columns[i]: float(input_values[i]) for i in range(len(columns))}
    df = pd.DataFrame([input_data])

    if model_type == "regression":
        model, scaler = logreg_model, logreg_scaler
    elif model_type == "random_forest":
        model, scaler = rf_model, None
    elif model_type == "xgboost":
        model, scaler = xgb_model, None
    else:
        model, scaler = dl_model, logreg_scaler

    try:
        if model_type in ["regression", "deep_learning"] and scaler is not None:
            df_processed = scaler.transform(df)
        else:
            df_processed = df

        if model_type == "deep_learning":
            predicted_probs = model.predict(df_processed)
            predicted_class = np.argmax(predicted_probs, axis=1)[0]
        else:
            predicted_class = model.predict(df_processed)[0]

        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        predicted_grade = grade_map.get(predicted_class, "Unknown")
        return f"Predicted Grade: {predicted_grade}"
    except Exception as e:
        return f"Error making prediction: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Fallback to 8050 locally
    app.run_server(debug=False, host="0.0.0.0", port=port)
