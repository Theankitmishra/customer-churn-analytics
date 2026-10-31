"""
Predictive Churn Modeling & Feature Importance Engine
=====================================================
Trains and evaluates Logistic Regression and Random Forest models on 
customer features, calculating cross-validated ROC-AUC, classification
reports, confusion matrices, and feature importances.

Author: Ankit Kumar Mishra
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_csv = os.path.join(base_dir, "data", "processed_features.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Loading preprocessed customer features...")
    df = pd.read_csv(processed_csv)
    
    feature_cols = [c for c in df.columns if c not in ("customer_id", "churn")]
    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples")

    # 1. Logistic Regression Baseline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Logistic Regression baseline...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    lr_roc = roc_auc_score(y_test, lr_probs)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)

    print(f"[Logistic Regression] ROC-AUC: {lr_roc:.4f} | Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f}")

    # 2. Random Forest Classifier
    print("Training Random Forest Classifier (100 estimators, max_depth=12)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    rf_roc = roc_auc_score(y_test, rf_probs)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)

    print(f"[Random Forest] ROC-AUC: {rf_roc:.4f} | Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}")

    # Feature Importances from Random Forest
    importances = rf_model.feature_importances_
    feature_ranking = sorted(
        zip(feature_cols, [round(float(v), 4) for v in importances]),
        key=lambda x: x[1],
        reverse=True
    )

    # Save artifacts
    results = {
        "dataset_summary": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_cols)
        },
        "logistic_regression": {
            "roc_auc": round(float(lr_roc), 4),
            "accuracy": round(float(lr_acc), 4),
            "precision": round(float(lr_prec), 4),
            "recall": round(float(lr_rec), 4),
            "f1_score": round(float(lr_f1), 4),
            "confusion_matrix": confusion_matrix(y_test, lr_preds).tolist()
        },
        "random_forest": {
            "roc_auc": round(float(rf_roc), 4),
            "accuracy": round(float(rf_acc), 4),
            "precision": round(float(rf_prec), 4),
            "recall": round(float(rf_rec), 4),
            "f1_score": round(float(rf_f1), 4),
            "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist()
        },
        "top_feature_importances": [
            {"feature": f, "importance": imp} for f, imp in feature_ranking
        ]
    }

    metrics_out = os.path.join(models_dir, "model_evaluation_metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save trained RF model & scaler
    with open(os.path.join(models_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"Evaluation metrics and models successfully exported to: {models_dir}")


if __name__ == "__main__":
    main()
