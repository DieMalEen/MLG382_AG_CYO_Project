import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
    x_train = train.drop(columns=["prognosis"])
    y_train = train["prognosis"] #Dependent variable for "training"

    x_test = test.drop(columns=["prognosis"])
    y_test = test["prognosis"] #Dependent variable for testing accuracy

    return x_train, y_train, x_test, y_test


def run_logistic_regression(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    #print(x_train_scaled)

    model = LogisticRegression(
        max_iter=100, 
        solver='saga',
        verbose=1,
        n_jobs=-1,
        warm_start=True)     # Train model

    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", round(accuracy, 4))

    target_names = np.unique(y_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    with open("artifacts/regression_model.pkl", "wb") as f:     # Save the model into a pkl file
        pickle.dump(model, f)

    with open('artifacts/regression_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return y_pred

def logistic_regression_graph(x_train, y_train, x_test, y_test):
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    # Train model
    model = LogisticRegression(solver='liblinear', warm_start=True)

    accuracies = []

    for i in range(1, 101):
        model.max_iter = i
        model.fit(x_train_scaled, y_train)

        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        accuracies.append(accuracy)

    plt.plot(range(1, 101), accuracies, marker='o', color='b')
    plt.title("Accuracy per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def run_random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(
        #n_estimators=50, 
        #max_depth= 10, 
        #max_features= 'sqrt', 
        n_jobs=1, 
        random_state=50)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\nRandom Forest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred))

    with open("artifacts/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

def run_xgboost(x_train, y_train, x_test, y_test):
    # Encode target labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    model = XGBClassifier(
        eval_metric='mlogloss', 
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.1,
        random_state=50
    )

    # Train the model
    model.fit(x_train, y_train_encoded)

    # Predict and decode labels
    y_pred_encoded = model.predict(x_test)
    y_pred = le.inverse_transform(y_pred_encoded)
    
    print("\nXGBoost Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_pred))

    with open("artifacts/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

def save_predictions(test, y_pred, model_name): # Predict and save models test_data.csv predictions
    df = test.copy()
    df[f"Predicted_{model_name}"] = y_pred
    df["Match"] = df["prognosis"] == df[f"Predicted_{model_name}"]
    df["Match"] = df["Match"].apply(lambda x: "True" if x else "False")
    df = df.sort_values(by="Match").reset_index(drop=True)
    df.to_csv(f"artifacts/{model_name}_predictions.csv", index=False)

def main():
    train_data, test_data = load_data()
    x_train, y_train, x_test, y_test = prepare_data(train_data, test_data)

    #logistic_regression_graph(x_train, y_train, x_test, y_test)

    y_pred_logreg = run_logistic_regression(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_logreg, "regression")

    y_pred_rf = run_random_forest(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_rf, "random_forest")

    y_pred_xgb = run_xgboost(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_xgb, "xgboost")
    
if __name__ == "__main__":
    main()