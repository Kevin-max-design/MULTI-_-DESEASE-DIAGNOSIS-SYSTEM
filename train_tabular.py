import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_breast_cancer_model(model_dir):
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Note: 0 is malignant, 1 is benign in sklearn's dataset
    # We'll map them later or keep standard.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest for Breast Cancer...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Breast Cancer Model Accuracy:", accuracy_score(y_test, y_pred))
    
    # Save model and feature names
    model_path = os.path.join(model_dir, "breast_cancer_model.pkl")
    joblib.dump({
        "model": model,
        "feature_names": data.feature_names.tolist(),
        "target_names": data.target_names.tolist() # ['malignant' 'benign']
    }, model_path)
    print(f"Saved to {model_path}\n")


def train_diabetes_model(model_dir):
    print("Loading Diabetes dataset...")
    # Fetching Pima Indians Diabetes database from OpenML
    # id 37 is standard Pima diabetes dataset
    diabetes = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
    df = diabetes.frame
    
    # Target is 'class', values 'tested_negative' and 'tested_positive'
    X = df.drop(columns=['class'])
    
    # Map target to 0 and 1
    y = df['class'].map({'tested_negative': 0, 'tested_positive': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest for Diabetes...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Diabetes Model Accuracy:", accuracy_score(y_test, y_pred))
    
    # Save model
    model_path = os.path.join(model_dir, "diabetes_model.pkl")
    joblib.dump({
        "model": model,
        "feature_names": X.columns.tolist(),
        "target_names": ['Negative', 'Positive']
    }, model_path)
    print(f"Saved to {model_path}\n")


if __name__ == "__main__":
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_breast_cancer_model(MODEL_DIR)
    train_diabetes_model(MODEL_DIR)
    
    print("All tabular models trained successfully.")
