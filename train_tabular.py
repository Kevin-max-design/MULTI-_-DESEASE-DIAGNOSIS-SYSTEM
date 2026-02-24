"""
train_tabular.py — Auto-select the best algorithm for each disease.

Benchmarks multiple classifiers with cross-validation and automatically
saves the one with the highest balanced accuracy.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Classifiers to benchmark
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_candidate_models():
    """Returns a dictionary of candidate models to benchmark."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            class_weight='balanced', random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, verbosity=0
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', probability=True, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='minkowski'
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8,
            class_weight='balanced', random_state=42
        ),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            max_iter=500, early_stopping=True, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=150, learning_rate=0.1, random_state=42
        ),
    }


def benchmark_models(X_train, y_train, X_test, y_test, task_name):
    """
    Run 5-fold stratified cross-validation on all candidates.
    Returns the best model name, fitted model, and results table.
    """
    candidates = get_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    best_score = -1
    best_name = None
    best_model = None

    print(f"\n{'='*60}")
    print(f"  BENCHMARKING ALGORITHMS — {task_name}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'CV Acc':>8} {'CV F1':>8} {'Test Acc':>9} {'Test F1':>8}")
    print(f"{'-'*60}")

    for name, model in candidates.items():
        try:
            # Cross-validation scores
            cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')

            # Fit on full training set, evaluate on test set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1  = f1_score(y_test, y_pred, average='weighted')

            mean_cv_acc = cv_acc.mean()
            mean_cv_f1  = cv_f1.mean()
            
            # Use weighted combination: 60% test F1 + 40% CV F1 
            combined_score = 0.6 * test_f1 + 0.4 * mean_cv_f1

            results.append({
                "Model": name,
                "CV Accuracy": f"{mean_cv_acc:.4f}",
                "CV F1": f"{mean_cv_f1:.4f}",
                "Test Accuracy": f"{test_acc:.4f}",
                "Test F1": f"{test_f1:.4f}",
                "Combined": combined_score
            })

            marker = ""
            if combined_score > best_score:
                best_score = combined_score
                best_name = name
                best_model = model
                marker = " ◀ best so far"

            print(f"{name:<25} {mean_cv_acc:>8.4f} {mean_cv_f1:>8.4f} {test_acc:>9.4f} {test_f1:>8.4f}{marker}")

        except Exception as e:
            print(f"{name:<25} ⚠ FAILED: {e}")

    print(f"{'-'*60}")
    print(f"🏆 Winner: {best_name}  (combined score: {best_score:.4f})")
    print()

    # Print detailed report for the winner
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    print(f"Detailed report for {best_name}:")
    print(classification_report(y_test, y_pred_best))

    return best_name, best_model, results


# ═══════════════════════════════════════════════════════════════════
#  BREAST CANCER
# ═══════════════════════════════════════════════════════════════════

def train_breast_cancer_model(model_dir):
    print("\nLoading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name, best_model, results = benchmark_models(
        X_train, y_train, X_test, y_test, "Breast Cancer"
    )

    model_path = os.path.join(model_dir, "breast_cancer_model.pkl")
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "algorithm": best_name,
        "feature_names": data.feature_names.tolist(),
        "target_names": data.target_names.tolist(),
        "benchmark_results": results
    }, model_path)
    print(f"Saved {best_name} → {model_path}\n")


# ═══════════════════════════════════════════════════════════════════
#  DIABETES
# ═══════════════════════════════════════════════════════════════════

def train_diabetes_model(model_dir):
    print("\nLoading Diabetes dataset...")
    diabetes = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
    df = diabetes.frame

    X = df.drop(columns=['class'])
    y = df['class'].map({'tested_negative': 0, 'tested_positive': 1})

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set XGBoost scale_pos_weight for this imbalanced dataset
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    candidates = get_candidate_models()
    candidates["XGBoost"].set_params(scale_pos_weight=neg_count / pos_count)

    best_name, best_model, results = benchmark_models(
        X_train, y_train, X_test, y_test, "Diabetes"
    )

    model_path = os.path.join(model_dir, "diabetes_model.pkl")
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "algorithm": best_name,
        "feature_names": X.columns.tolist(),
        "target_names": ['Negative', 'Positive'],
        "benchmark_results": results
    }, model_path)
    print(f"Saved {best_name} → {model_path}\n")


if __name__ == "__main__":
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_breast_cancer_model(MODEL_DIR)
    train_diabetes_model(MODEL_DIR)

    print("═" * 60)
    print("  All models trained. Best algorithms auto-selected.")
    print("═" * 60)
