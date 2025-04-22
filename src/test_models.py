import pickle
import pandas as pd
import numpy as np

# Example input string from user
student_input = "1136,15,0,3,1,10.531898851795788,12,0,2,1,0,0,0,2.122638529628868,3.0"

def load_model_and_scaler(model_type):
    model_path = f"artifacts/{model_type}_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    scaler = None
    if model_type in ["regression", "deep_learning"]:
        with open("artifacts/regression_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    return model, scaler

def preprocess_input(input_string, model_type, scaler):
    # Define the feature columns (excluding StudentID, GPA, and GradeClass)
    columns = ["Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly", "Absences", 
               "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"]
    
    # Split and slice the input (remove ID, GPA, GradeClass)
    values = input_string.split(',')[1:-2]
    input_data = {columns[i]: float(values[i]) for i in range(len(columns))}
    
    df = pd.DataFrame([input_data])

    # Scale only for regression and xgboost (which was scaled during training)
    if model_type in ["regression", "deep_learning"] or model_type == "xgboost":
        _, scaler = load_model_and_scaler("regression")
        df_scaled = scaler.transform(df)
        return df_scaled
    return df  # Random Forest uses raw features

def predict_grade(input_string, model_type="regression"):
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_type)

    # Preprocess input
    input_processed = preprocess_input(input_string, model_type, scaler)

    # Predict
    if model_type == "deep_learning":
        # Deep Learning model outputs probabilities, so take the class with the highest probability
        predicted_probs = model.predict(input_processed)
        predicted_class = np.argmax(predicted_probs, axis=1)[0]
    else:
        # Other models directly predict the class
        predicted_class = model.predict(input_processed)[0]

    # Map class to letter grade
    grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    return grade_map[predicted_class]

# Example usage
print("Predicted grade with Logistic Regression:", predict_grade(student_input, "regression"))
print("Predicted grade with Random Forest:", predict_grade(student_input, "random_forest"))
print("Predicted grade with XGBoost:", predict_grade(student_input, "xgboost"))
print("Predicted grade with Deep Learning:", predict_grade(student_input, "deep_learning"))