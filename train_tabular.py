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
    diabetes = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
    df = diabetes.frame
    
    X = df.drop(columns=['class'])
    y = df['class'].map({'tested_negative': 0, 'tested_positive': 1})
    
    # Feature scaling for better model performance
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate class weight ratio for imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print("Training XGBoost for Diabetes...")
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Diabetes Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    model_path = os.path.join(model_dir, "diabetes_model.pkl")
    joblib.dump({
        "model": model,
        "scaler": scaler,
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
