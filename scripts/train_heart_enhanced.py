"""
Enhanced Heart Risk Model Training
===================================
Trains multiple classifiers with hyperparameter tuning on the UCI Heart Disease dataset.
Compares XGBoost, LightGBM, Random Forest, Gradient Boosting, SVM, Logistic Regression.
Saves the best model by AUC-ROC.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Load data
data_path = PROJECT_ROOT / "data" / "vitals" / "heart.csv"
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} records from heart.csv")
print(f"Class distribution:\n{df['target'].value_counts()}")

X = df.drop("target", axis=1).values
y = df["target"].values

# Check imbalance
from collections import Counter
print(f"Class counts: {Counter(y)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(f"After SMOTE: {Counter(y_resampled)}")

# Define classifiers with param grids
classifiers = {
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
    ),
    "LightGBM": (
        LGBMClassifier(random_state=42, verbose=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=5000, random_state=42),
        {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
        }
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_score = 0
best_model = None
best_name = ""

print("\n" + "=" * 70)
print("Training and Evaluating Models (5-fold Stratified CV)")
print("=" * 70)

for name, (clf, param_grid) in classifiers.items():
    print(f"\n--- {name} ---")
    grid = GridSearchCV(
        clf, param_grid, cv=cv, scoring="roc_auc",
        n_jobs=-1, verbose=0, refit=True
    )
    grid.fit(X_resampled, y_resampled)

    # Evaluate on the full resampled data with best estimator
    best_est = grid.best_estimator_
    y_pred = best_est.predict(X_resampled)
    y_proba = best_est.predict_proba(X_resampled)[:, 1]

    acc = accuracy_score(y_resampled, y_pred)
    f1 = f1_score(y_resampled, y_pred, average="weighted")
    auc = roc_auc_score(y_resampled, y_proba)

    # Also evaluate on original (unaugmented) data
    y_pred_orig = best_est.predict(X_scaled)
    y_proba_orig = best_est.predict_proba(X_scaled)[:, 1]
    acc_orig = accuracy_score(y, y_pred_orig)
    f1_orig = f1_score(y, y_pred_orig, average="weighted")
    auc_orig = roc_auc_score(y, y_proba_orig)

    print(f"  Best params: {grid.best_params_}")
    print(f"  CV AUC-ROC: {grid.best_score_:.4f}")
    print(f"  Original data — Accuracy: {acc_orig:.4f}  F1: {f1_orig:.4f}  AUC: {auc_orig:.4f}")

    results.append({
        "Model": name,
        "CV AUC": round(grid.best_score_, 4),
        "Accuracy": round(acc_orig, 4),
        "F1": round(f1_orig, 4),
        "AUC-ROC": round(auc_orig, 4),
    })

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = best_est
        best_name = name

# Print comparison table
print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\nBest model: {best_name} (CV AUC-ROC: {best_score:.4f})")

# Final evaluation on original data
y_pred_final = best_model.predict(X_scaled)
final_acc = accuracy_score(y, y_pred_final)
print(f"\n{'='*70}")
print(f"ACCURACY COMPARISON")
print(f"{'='*70}")
print(f"Old accuracy (Random Forest baseline): 83.00%")
print(f"New accuracy ({best_name}):             {final_acc*100:.2f}%")
print(f"Improvement:                            +{(final_acc*100 - 83):.2f}%")

# Save best model and scaler
model_path = MODELS_DIR / "heart_risk.joblib"
scaler_path = MODELS_DIR / "heart_risk_scaler.joblib"
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nSaved: {model_path}")
print(f"Saved: {scaler_path}")
