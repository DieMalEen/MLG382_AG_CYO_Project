import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np

def load_data():
    train = pd.read_csv("data/train_data.csv", delimiter=",")
    test = pd.read_csv("data/test_data.csv", delimiter=",")
    
    return train, test

def prepare_data(train, test):  
    x_train = train.drop(columns=["TempCategory", "RainTomorrow", "DayAvgTemp", "MinTemp", "MaxTemp", "Temp3pm", "Temp9am"])
    y_train = train[["RainTomorrow", "TempCategory"]]  # Include both targets

    x_test = test.drop(columns=["TempCategory", "RainTomorrow", "DayAvgTemp", "MinTemp", "MaxTemp", "Temp3pm", "Temp9am"])
    y_test = test[["RainTomorrow", "TempCategory"]]  # Include both targets

    # Combine train and test for consistent encoding
    combined = pd.concat([x_train, x_test], axis=0)

    # Encode categorical features
    for col in combined.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back
    x_train = combined.iloc[:len(x_train), :].copy()
    x_test = combined.iloc[len(x_train):, :].copy()

    # Encode target columns
    for col in y_train.columns:
        le = LabelEncoder()
        y_train[col] = le.fit_transform(y_train[col].astype(str))
        y_test[col] = le.transform(y_test[col].astype(str))

    return x_train, y_train, x_test, y_test

def run_logistic_regression(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {}
    y_preds = {}

    for col in y_train.columns:
        print(f"\nTraining Logistic Regression for target: {col}")
        model = LogisticRegression(
            max_iter=100, 
            solver='saga',
            verbose=1,
            n_jobs=-1,
            warm_start=True
        )

        model.fit(x_train_scaled, y_train[col])
        y_pred = model.predict(x_test_scaled)
        y_preds[col] = y_pred

        print(f"\nLogistic Regression Accuracy ({col}): {round(accuracy_score(y_test[col], y_pred), 4)}")
        print(f"\nClassification Report{col}:\n{classification_report(y_test[col], y_pred)}")

        models[col] = model

    # Save models and scaler
    with open("artifacts/logistic_regression_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("artifacts/logistic_regression_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Combine predictions
    y_pred_df = pd.DataFrame(y_preds)
    return y_pred_df
    
def run_random_forest(x_train, y_train, x_test, y_test):
    models = {}
    y_preds = {}

    for col in y_train.columns:
        print(f"\nTraining Random Forest for target: {col}")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=50
        )

        # Fit model on features and the specific target column
        model.fit(x_train, y_train[col])
        y_pred = model.predict(x_test)
        y_preds[col] = y_pred

        # Evaluation metrics
        print(f"\nRandom Forest Accuracy ({col}): {round(accuracy_score(y_test[col], y_pred), 4)}")
        print(f"\nClassification Report{col}:\n{classification_report(y_test[col], y_pred)}")

        models[col] = model

    # Save models dictionary to artifacts/random_forest_models.pkl
    with open("artifacts/random_forest_models.pkl", "wb") as f:
        pickle.dump(models, f)

    # Combine predictions into a DataFrame and return
    y_pred_df = pd.DataFrame(y_preds)
    return y_pred_df


def run_xgboost(x_train, y_train, x_test, y_test):
    preds = []

    for col in y_train.columns:
        model = XGBClassifier(
            eval_metric='mlogloss', 
            n_estimators=200, 
            max_depth=10, 
            learning_rate=0.1,
            random_state=50
        )

        model.fit(x_train, y_train[col])
        pred = model.predict(x_test)
        preds.append(pred)

        with open(f"artifacts/xgboost_model_{col}.pkl", "wb") as f:
            pickle.dump(model, f)

        print(f"\nXGBoost Accuracy ({col}):", round(accuracy_score(y_test[col], pred), 4))
        print(f"\nClassification Report:")
        print(classification_report(y_test[col], pred))

    y_pred = np.vstack(preds).T
    return y_pred

def save_predictions(test, y_pred, model_name):
    df = test.copy()

    # Convert NumPy array to DataFrame if needed
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=["RainTomorrow", "TempCategory"], index=df.index)

    # Map numerical predictions back to strings
    rain_map = {0: "no", 1: "yes"}
    temp_map = {0: "very cold", 1: "cold", 2: "medium", 3: "warm", 4: "very warm"}

    y_pred["RainTomorrow"] = y_pred["RainTomorrow"].map(rain_map)
    y_pred["TempCategory"] = y_pred["TempCategory"].map(temp_map)

    # Normalize actual values to strings for accurate comparison
    df["RainTomorrow"] = df["RainTomorrow"].astype(str).str.strip().str.lower()
    df["TempCategory"] = df["TempCategory"].astype(str).str.strip().str.lower()
    y_pred["RainTomorrow"] = y_pred["RainTomorrow"].astype(str).str.strip().str.lower()
    y_pred["TempCategory"] = y_pred["TempCategory"].astype(str).str.strip().str.lower()

    # Add predictions to dataframe
    df[f"Predicted_{model_name}_RainTomorrow"] = y_pred["RainTomorrow"]
    df[f"Predicted_{model_name}_TempCategory"] = y_pred["TempCategory"]

    # Calculate matches
    df["Match_RainTomorrow"] = df["RainTomorrow"] == df[f"Predicted_{model_name}_RainTomorrow"]
    df["Match_TempCategory"] = df["TempCategory"] == df[f"Predicted_{model_name}_TempCategory"]

    # Convert matches to "True"/"False"
    df["Match_RainTomorrow"] = df["Match_RainTomorrow"].apply(lambda x: "True" if x else "False")
    df["Match_TempCategory"] = df["Match_TempCategory"].apply(lambda x: "True" if x else "False")

    # Keep only the needed columns
    output_columns = [
        "RainTomorrow", "TempCategory",
        f"Predicted_{model_name}_RainTomorrow",
        f"Predicted_{model_name}_TempCategory",
        "Match_RainTomorrow", "Match_TempCategory"
    ]
    df = df[output_columns]

    # Save to CSV
    df.to_csv(f"artifacts/{model_name}_predictions.csv", index=False)

def main():
    train_data, test_data = load_data()
    x_train, y_train, x_test, y_test = prepare_data(train_data, test_data)

    y_pred_logreg = run_logistic_regression(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_logreg, "regression")

    y_pred_rf = run_random_forest(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_rf, "random_forest")

    y_pred_xgb = run_xgboost(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_xgb, "xgboost")
    
if __name__ == "__main__":
    main()