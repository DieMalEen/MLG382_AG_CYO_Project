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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
    rain_enc = LabelEncoder()
    temp_enc = LabelEncoder()
    y_train["RainTomorrow"] = rain_enc.fit_transform(y_train["RainTomorrow"].astype(str))
    y_train["TempCategory"] = temp_enc.fit_transform(y_train["TempCategory"].astype(str))
    y_test["RainTomorrow"] = rain_enc.transform(y_test["RainTomorrow"].astype(str))
    y_test["TempCategory"] = temp_enc.transform(y_test["TempCategory"].astype(str))

    return x_train, y_train, x_test, y_test, rain_enc, temp_enc

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

def preprocess_for_deep_learning(x_train, y_train, x_test, y_test):
    # Use LabelEncoder instead of one-hot encoding, matching the other models
    cat_cols = x_train.select_dtypes(include="object").columns
    combined = pd.concat([x_train, x_test], axis=0)
    
    # Dictionary to store LabelEncoders for categorical features
    label_encoders = {}
    
    # Encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        label_encoders[col] = le
    
    # Split back
    x_train = combined.iloc[:len(x_train), :].copy()
    x_test = combined.iloc[len(x_train):, :].copy()

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    # Encode target columns
    rain_encoder = LabelEncoder()
    temp_encoder = LabelEncoder()
    y_train_rain = rain_encoder.fit_transform(y_train["RainTomorrow"])
    y_train_temp = temp_encoder.fit_transform(y_train["TempCategory"])
    y_test_rain = rain_encoder.transform(y_test["RainTomorrow"])
    y_test_temp = temp_encoder.transform(y_test["TempCategory"])

    y_train_combined = torch.tensor(list(zip(y_train_rain, y_train_temp)), dtype=torch.long)
    y_test_combined = torch.tensor(list(zip(y_test_rain, y_test_temp)), dtype=torch.long)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_combined)
    test_dataset = TensorDataset(X_test_tensor, y_test_combined)

    # Save the label encoders for categorical features
    with open("artifacts/deep_learning_label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    return train_dataset, test_dataset, X_train_tensor.shape[1], rain_encoder, temp_encoder, scaler

# === Neural Network ===
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

def build_deep_learning_model(input_dim, rain_classes, temp_classes):
    return DualOutputNN(input_dim, rain_classes, temp_classes)

def train_deep_learning_model(model, train_dataset, epochs=100, batch_size=32, learning_rate=0.001):
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            y_rain, y_temp = targets[:, 0], targets[:, 1]
            optimizer.zero_grad()
            out_rain, out_temp = model(inputs)
            loss_rain = criterion(out_rain, y_rain)
            loss_temp = criterion(out_temp, y_temp)
            loss = loss_rain + loss_temp
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

def evaluate_deep_learning_model(model, test_dataset, rain_encoder, temp_encoder):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=32)
    all_rain_preds, all_temp_preds = [], []
    all_rain_true, all_temp_true = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            y_rain, y_temp = targets[:, 0], targets[:, 1]
            out_rain, out_temp = model(inputs)
            rain_preds = torch.argmax(out_rain, dim=1)
            temp_preds = torch.argmax(out_temp, dim=1)
            all_rain_preds.extend(rain_preds.numpy())
            all_temp_preds.extend(temp_preds.numpy())
            all_rain_true.extend(y_rain.numpy())
            all_temp_true.extend(y_temp.numpy())

    # Debug: Print the encoder classes
    print("RainTomorrow classes:", rain_encoder.classes_)
    print("TempCategory classes:", temp_encoder.classes_)

    # Define meaningful target names
    rain_target_names = ["No", "Yes"]  # Adjust based on actual classes
    temp_target_names = ["Very Cold", "Cold", "Medium", "Warm", "Very Warm"]  # Adjust based on actual classes

    print("\n--- RainTomorrow ---")
    print("Accuracy:", accuracy_score(all_rain_true, all_rain_preds))
    print(classification_report(all_rain_true, all_rain_preds, target_names=rain_target_names))

    print("\n--- TempCategory ---")
    print("Accuracy:", accuracy_score(all_temp_true, all_temp_preds))
    print(classification_report(all_temp_true, all_temp_preds, target_names=temp_target_names))

    return list(zip(all_rain_preds, all_temp_preds))

def save_deep_learning_model(model, path="artifacts/weather_deep_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def add_temp_category(df):
    # Simple binning logic, adjust thresholds as needed
    bins = [-float("inf"), 10, 25, float("inf")]
    labels = ["Cold", "Mild", "Hot"]
    df["TempCategory"] = pd.cut(df["Temp9am"], bins=bins, labels=labels)
    return df

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
    x_train, y_train, x_test, y_test, rain_enc, temp_enc = prepare_data(train_data, test_data)
    
    # Save LabelEncoders for target variables
    with open("artifacts/rain_encoder.pkl", "wb") as f:
        pickle.dump(rain_enc, f)
    with open("artifacts/temp_encoder.pkl", "wb") as f:
        pickle.dump(temp_enc, f)

    y_pred_logreg = run_logistic_regression(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_logreg, "regression")

    y_pred_rf = run_random_forest(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_rf, "random_forest")

    y_pred_xgb = run_xgboost(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_xgb, "xgboost")
    
    train_ds, test_ds, input_dim, rain_enc, temp_enc, dl_scaler = preprocess_for_deep_learning(x_train, y_train, x_test, y_test)

    # Save the deep learning scaler
    with open("artifacts/deep_learning_scaler.pkl", "wb") as f:
        pickle.dump(dl_scaler, f)

    model = build_deep_learning_model(input_dim, len(rain_enc.classes_), len(temp_enc.classes_))
    train_deep_learning_model(model, train_ds)
    raw_preds = evaluate_deep_learning_model(model, test_ds, rain_enc, temp_enc)
    y_pred_dpl = pd.DataFrame(raw_preds, columns=["RainTomorrow", "TempCategory"])

    save_deep_learning_model(model)
    save_predictions(test_data, y_pred_dpl, "deep_learning")
    
if __name__ == "__main__":
    main()